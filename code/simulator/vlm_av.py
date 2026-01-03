from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

from PIL import Image


@dataclass
class VLMActionStep:
    """单步动作：用于在 CARLA 中做“运动学回放/跟踪”的最小动作表示。"""

    dx: float  # 前向位移（m）
    dy: float  # 横向位移（m）：**与 main.py 的 make_transform 输入一致**（正负方向以 main.py 现有逻辑为准）
    dyaw: float  # 航向增量（rad），当前主循环未使用
    reason: str = ""  # 模型解释（可选，用于 UI 显示/调试）


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _to_data_url(pil_img: Image.Image, max_side: int = 512, quality: int = 80) -> str:
    """PIL.Image -> data:image/jpeg;base64,...（自动等比缩放以降低延迟与成本）。"""
    img = pil_img
    w, h = img.size
    scale = max(w, h) / float(max_side) if max(w, h) > max_side else 1.0
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    从模型输出中提取 JSON object。
    兼容：
    - 纯 JSON
    - ```json ... ```
    - 文本中夹杂一段 {...}
    """
    if not text:
        return None

    # 去掉代码围栏
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 优先：整体就是 JSON
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 次选：抽取第一段 {...}
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class OpenAICompatibleVLMLivePolicy:
    """
    用“OpenAI Chat Completions 兼容接口”的 VLM 做在线决策。

    适配：
    - OpenAI 官方
    - vLLM / LMDeploy / SGLang 等提供的 OpenAI-compatible /v1/chat/completions

    设计目标：
    - **高层决策**交给 VLM（看图 + 少量状态），
    - **低层执行**保持简单、可控（在 main.py 里做限幅与回退）。
    """

    def __init__(
        self,
        api_base: str | None,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 20.0,
        min_interval_s: float = 0.5,
        max_image_side: int = 512,
        memory_steps: int = 0,
    ):
        self.api_base = (api_base or "").rstrip("/") or None
        self.model = model.strip()
        self.api_key = api_key.strip() or "EMPTY"
        self.timeout_s = float(timeout_s)
        self.min_interval_s = float(min_interval_s)
        self.max_image_side = int(max_image_side)
        self.memory_steps = int(max(0, memory_steps))

        self._last_call_t = 0.0
        self._last_step: Optional[VLMActionStep] = None
        self._client = None
        self._memory: list[str] = []

    @staticmethod
    def _default_base_url_for_model(model_name: str) -> str | None:
        """
        参考你给的 OpenEQA/infer.py 的端口映射。
        如果 model_name 不在映射中，返回 None（要求外部显式提供 base_url）。
        """
        mapping = {
            "qwen_7b": "http://localhost:5008/v1",
            "vebrain": "http://localhost:5003/v1",
            "cosmos_7b": "http://localhost:5001/v1",
            "robobrain2_7b": "http://localhost:5030/v1",
            "llava-onevision": "http://localhost:5002/v1",
            "internvl2_5": "http://localhost:5013/v1",
            "Qwen/Qwen3-VL-2B-Instruct-FP8": "http://localhost:8000/v1",
            "ep-20251207152729-9s7fn":"https://ark.cn-beijing.volces.com/api/v3", #fc0b204d-f97e-45ec-81f8-dfc669b62273
            "ep-20251228182830-qcsnj":"https://ark.cn-beijing.volces.com/api/v3", #fc0b204d-f97e-45ec-81f8-dfc669b62273
            "gpt-5.2":"https://yinli.one/v1",
        }
        return mapping.get(model_name)

    def _build_messages(
        self,
        images: list[Image.Image],
        speed_mps: float,
        lane: int,
    ) -> list[dict[str, Any]]:
        system = (
            "你是 CARLA 自动驾驶换道决策模型，这是一条4车道场景，输入包含鸟瞰图（从上往下看，图中间是你的车，上下左右是其他车辆，请注意区分）。\n"
            "你必须仅输出一个 JSON 对象，字段：\n" #### 前视图(车头视角，放在你车的正前方车头上，所以你前方的车就是别的车，不是你自己的车，请注意区分）、鸟瞰图（从上往下看，图中间是你的车，上下左右是其他车辆，请注意区分）
            "- dx: float，0.0\n"
            "- dy: float，横向位移（米），进行换道的操作，车辆向右换道时为正，向左换道时为负，保持直行则为0，推荐区间 [-1.0-1.0]\n"
            "- dyaw: float，航向增量（弧度，可为 0），进行换道的操作，推荐区间 [-0.01-0.01]\n"
            "- reason: string，简短说明\n"
            "场景：三车道高速路，1 AV + 5 HV，HV 车道固定，AV车速最快。\n"
            "目标：安全、高效完成行驶任务。\n"
            "输出只能是合法 JSON，不要额外文本或代码围栏。"
        )

        memory_text = ""
        if self.memory_steps > 0 and self._memory:
            recent = self._memory[-self.memory_steps :]
            memory_text = "最近决策记录（从旧到新）：\n" + "\n".join(f"- {x}" for x in recent) + "\n\n"

        user_text = (
            f"当前速度 speed_mps={speed_mps:.2f}，当前车道索引 lane_index={lane}（左=-1，中=0，右=+1）。目前图像显示4个车道，但最右边的车道为不可通行车道，所以（左=-1，中=0，右=+1）为左边3个车道\n"
            f"{memory_text}"
            "请根据鸟瞰图输出下一小步动作并返回 JSON 对象。\n" ###请根据前视图和鸟瞰图输出下一小步动作并返回 JSON 对象。
            "要求：\n"
            "- dx：一直为 0。\n"
            "- dy/dyaw：用于换道（向右换道 dy>0，向左换道 dy<0；保持直行则 dy=0）。\n"
            "- 若当前车道正前方有慢车，应保持前进并选择安全车道换道超越。\n"
            "- 注意不要与其他车碰撞，也不要驶出道路。\n"
            "并在 reason 中说明你判断当前车处于从左到右数的第几个车道，以及为何选择该动作。"
        )

        contents: list[dict[str, Any]] = []
        for im in images:
            contents.append({"type": "image_url", "image_url": {"url": _to_data_url(im, max_side=self.max_image_side)}})
        contents.append({"type": "text", "text": user_text})

        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                # 与 OpenEQA/infer.py 的 contents 组织方式一致：先 image_url 再 text
                "content": contents,
            },
        ]

    def _get_openai_client(self):
        """
        优先使用 openai SDK（与 OpenEQA/infer.py 一致）。
        若环境无 openai 包，则返回 None（后续走 requests 直连）。
        """
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None

        base_url = self.api_base or self._default_base_url_for_model(self.model)
        if not base_url:
            return None
        self._client = OpenAI(api_key=self.api_key, base_url=base_url)
        return self._client

    def step(
        self,
        front_pil_image: Image.Image | None,
        speed_mps: float,
        lane: int,
        top_pil_image: Image.Image | None ,
    ) -> Optional[VLMActionStep]:
        now = time.time()
        if self._last_step is not None and (now - self._last_call_t) < self.min_interval_s:
            return self._last_step
        if front_pil_image is None:
            return self._last_step

        images: list[Image.Image] = [front_pil_image]
        if top_pil_image is not None:
            images.append(top_pil_image)

        # 1) 优先 OpenAI SDK（与 OpenEQA/infer.py 调用方式一致）
        client = self._get_openai_client()
        if client is not None:
            resp = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(
                    images=images,
                    speed_mps=speed_mps,
                    lane=lane,
                ),
                temperature=0.0,
                max_tokens=256,
            )
            content = resp.choices[0].message.content or ""
        else:
            # 2) 回退：requests 直连 /v1/chat/completions
            try:
                import requests  # noqa: WPS433
            except Exception as e:
                raise RuntimeError("vlm_av: missing dependency 'openai' and 'requests'") from e

            base_url = self.api_base or self._default_base_url_for_model(self.model)
            if not base_url:
                raise RuntimeError(
                    "vlm_av: cannot infer VLM base_url; set VLM_API_BASE (e.g. http://localhost:8000/v1)"
                )
            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": self._build_messages(
                    images=images,
                    speed_mps=speed_mps,
                    lane=lane,
                ),
                "temperature": 0.0,
                "max_tokens": 512,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            
        print(content)
        obj = _extract_json_object(str(content))
        if not obj:
            return self._last_step

        try:
            dx = float(obj.get("dx", 0.0))
            dy = float(obj.get("dy", 0.0))
            dyaw = float(obj.get("dyaw", 0.0))
            reason = str(obj.get("reason", "") or "")
        except Exception:
            return self._last_step

        # 最终安全限幅（主循环还会再做一层限幅）
        step = VLMActionStep(
            dx=_clamp(dx, 0.0, 3.0),
            dy=_clamp(dy, -1.5, 1.5),
            dyaw=_clamp(dyaw, -0.2, 0.2),
            reason=reason,
        )
        self._last_call_t = now
        self._last_step = step
        if self.memory_steps > 0:
            self._memory.append(
                f'dx={step.dx:+.2f}, dy={step.dy:+.2f}, dyaw={step.dyaw:+.3f}, reason={step.reason}'
            )
            keep = max(self.memory_steps * 2, 10)
            if len(self._memory) > keep:
                self._memory = self._memory[-keep:]
        return step

