from __future__ import annotations

import argparse
import carla
import json
import os
import queue
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

import numpy as np
from PIL import Image

try:
    # 可选：DriveVLA 回放策略（用于把 DriveVLA 推理输出接到 AV）
    from drivevla_av import DriveVLAReplayPolicy, DriveVLALivePolicy
except Exception:
    DriveVLAReplayPolicy = None  # type: ignore
    DriveVLALivePolicy = None  # type: ignore

try:
    # 可选：通用多模态大模型（OpenAI-compatible VLM）在线决策
    from vlm_av import OpenAICompatibleVLMLivePolicy, VLMActionStep
except Exception:
    OpenAICompatibleVLMLivePolicy = None  # type: ignore
    VLMActionStep = None  # type: ignore


def _str2bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "t", "yes", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CARLA AV scenario with optional VLM/DriveVLA control."
    )
    parser.add_argument(
        "--av-mode",
        default=os.environ.get("AV_MODE", "vlm_live").strip() or "vlm_live",
        choices=["cruise", "drivevla_replay", "drivevla_live", "vlm_live"],
        help="选择 AV 控制模式（默认 vlm_live，或 cruise/drivevla_live/vlm_live）。",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("VLM_MODEL", "").strip() or "Qwen/Qwen3-VL-2B-Instruct-FP8",
        help="VLM 模型名（OpenAI-compatible）。",
    )
    parser.add_argument(
        "--vlm-api-base",
        default=os.environ.get("VLM_API_BASE", "").strip() or "http://localhost:8000/v1",
        help="VLM 接口 base URL，例如 http://localhost:8000/v1。",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=os.environ.get("VLM_API_KEY", "EMPTY").strip() or "EMPTY",
        help="VLM API Key（默认 EMPTY）。",
    )
    parser.add_argument(
        "--vlm-use-top",
        type=int,
        default=1 if _str2bool(os.environ.get("VLM_USE_TOP", "0")) else 1,
        choices=[0, 1],
        help="是否把鸟瞰图也送入 VLM（0/1）。",
    )
    parser.add_argument(
        "--print-av-action",
        type=int,
        default=1 if _str2bool(os.environ.get("PRINT_AV_ACTION", "0")) else 1,
        choices=[0, 1],
        help="是否打印 AV 动作日志（0/1）。",
    )
    parser.add_argument(
        "--vlm-timeout-s",
        type=float,
        default=float(os.environ.get("VLM_TIMEOUT_S", "2")),
        help="VLM 请求超时时间（秒）。",
    )
    parser.add_argument(
        "--vlm-min-interval-s",
        type=float,
        default=float(os.environ.get("VLM_MIN_INTERVAL_S", "0.1")),
        help="VLM 最小调用间隔（秒）。",
    )
    parser.add_argument(
        "--vlm-max-image-side",
        type=int,
        default=int(os.environ.get("VLM_MAX_IMAGE_SIDE", "512")),
        help="送入 VLM 的图像最大边（像素）。",
    )
    parser.add_argument(
        "--vlm-memory-steps",
        type=int,
        default=int(os.environ.get("VLM_MEMORY_STEPS", "3")),
        help="VLM 记忆步数：把最近 N 步决策文本拼进下一次提示（0 表示关闭）。",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=int(os.environ.get("HTTP_PORT", "8080")),
        help="网页相机/状态面板端口（默认 8080）。端口冲突时可改为 8081 等。",
    )
    parser.add_argument(
        "--record-mp4",
        type=int,
        default=int(os.environ.get("RECORD_MP4", "0")),
        choices=[0, 1],
        help="是否录制并保存 mp4 到 output/（0/1）。默认 0。",
    )
    parser.add_argument(
        "--record-every-n",
        type=int,
        default=int(os.environ.get("RECORD_EVERY_N", "2")),
        help="每 N 帧写入一次视频（降低体积）。默认 2。",
    )
    parser.add_argument(
        "--record-max-side",
        type=int,
        default=int(os.environ.get("RECORD_MAX_SIDE", "640")),
        help="写入视频前将图像等比缩放到最大边（像素）。默认 640。",
    )
    parser.add_argument(
        "--record-include-top",
        type=int,
        default=int(os.environ.get("RECORD_INCLUDE_TOP", "1")),
        choices=[0, 1],
        help="mp4 是否合并鸟瞰图（0/1）。默认 1。",
    )
    return parser.parse_args()


def _resize_for_record(img: Image.Image, max_side: int) -> Image.Image:
    """等比缩放，用于录制时降低体积。"""
    w, h = img.size
    scale = max(w, h) / float(max_side) if max(w, h) > max_side else 1.0
    if scale > 1.0:
        return img.resize((int(w / scale), int(h / scale)), Image.BICUBIC)
    return img


def _pad_to_multiple(img: Image.Image, m: int = 16) -> Image.Image:
    """把图像 pad 到宽高均为 m 的倍数，避免编码器自动 resize。"""
    w, h = img.size
    new_w = int(np.ceil(w / m) * m)
    new_h = int(np.ceil(h / m) * m)
    if new_w == w and new_h == h:
        return img
    out = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    out.paste(img, (0, 0))
    return out


# HTTP 相机最新一帧 JPEG（前向视角 & 鸟瞰视角）
latest_front_jpeg = None
latest_top_jpeg = None
latest_speed_text = "No data yet"
latest_vlm_text = "No VLM data yet"
latest_lock = threading.Lock()
status_lock = threading.Lock()

# 车道/场景参数（作业要求）
LANE_WIDTH = 3.5
LANE_SHIFT = -LANE_WIDTH   # 全局右移一车道
LANE_RIGHT_Y = -LANE_WIDTH + LANE_SHIFT  # 右车道中心
LANE_MID_Y = 0.0 + LANE_SHIFT            # 中车道中心
LANE_LEFT_Y = LANE_WIDTH + LANE_SHIFT    # 左车道中心

# 车辆长度近似值（用于后续 IDM 等，先占位）
VEH_LENGTH = 4.5

# IDM 目标速度（120 km/h）
V_DESIRED = 130/3.6
# AV 初始速度（80 km/h）
AV_INIT_SPEED = 120 / 3.6

# 三种驾驶风格参数
IDM_STYLES = {
    "conservative": {"a_max": 2.0, "b": 2.0, "s0": 5.0, "T": 2.0},
    "normal": {"a_max": 4.0, "b": 2.5, "s0": 3.0, "T": 1.5},
    "aggressive": {"a_max": 6.0, "b": 3.0, "s0": 2.0, "T": 1.0},
}


def lane_index_from_d(d: float) -> int:
    """根据横向偏移(shifted-d)近似车道索引：左=+1，中=0，右=-1。"""
    return int(round(d / LANE_WIDTH))


class KinematicState:
    """只存储纵向位置/速度/车道，用于前车搜索。"""

    __slots__ = ("s", "v", "d", "lane")

    def __init__(self, s: float, v: float, d: float, lane: int):
        self.s = s
        self.v = v
        self.d = d  # shifted-d（用于 make_transform）
        self.lane = lane


class IDMVehicle:
    """基于 IDM 的简易纵向跟驰车辆（车道固定不变）。"""

    def __init__(
        self,
        name: str,
        actor: carla.Actor,
        s0: float,
        d0: float,
        v0: float,
        style_params: dict,
        v_desired: float = V_DESIRED,
    ):
        self.name = name
        self.actor = actor
        self.s = s0
        self.d = d0  # shifted-d（与 make_transform 入参一致）
        self.v = v0
        self.a = 0.0
        self.lane = lane_index_from_d(d0)
        self.params = style_params
        self.v_desired = v_desired

    def _compute_acc(self, front: KinematicState | "IDMVehicle" | None) -> float:
        """按 IDM 公式计算加速度。"""
        a_max = self.params["a_max"]
        v_ratio = self.v / self.v_desired if self.v_desired > 1e-6 else 0.0
        free_term = 1.0 - v_ratio**4

        if front is None:
            return a_max * free_term

        s_gap = max(front.s - self.s - VEH_LENGTH, 0.1)
        delta_v = self.v - front.v
        s_star = self.params["s0"] + self.v * self.params["T"]
        denom = max(2.0 * np.sqrt(max(a_max * self.params["b"], 1e-6)), 1e-6)
        s_star += (self.v * delta_v) / denom
        interact_term = (s_star / s_gap) ** 2
        a_idm = a_max * (free_term - interact_term)
        # 若速度超过期望值，增加柔性制动项将其拉回（不做硬截断）
        if self.v > self.v_desired:
            a_idm -= 0.6 * (self.v - self.v_desired)
        return a_idm

    def step(
        self,
        dt: float,
        front: KinematicState | "IDMVehicle" | None,
        to_transform,
    ) -> None:
        """离散积分更新位置，并把结果同步到 CARLA actor。"""
        if self.actor is None or not self.actor.is_alive:
            return

        a = self._compute_acc(front)
        self.a = a
        self.v = max(0.0, self.v + a * dt)
        self.s = self.s + self.v * dt + 0.5 * a * dt * dt

        try:
            self.actor.set_transform(to_transform(self.s, self.d))
        except RuntimeError:
            # 偶发的同步更新失败可忽略，下一帧会继续更新
            pass

# 选择用于构建直线路段的基础出生点索引（可根据需要调整）
BASE_SPAWN_INDEX = 100


class CameraHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 屏蔽默认的访问日志输出
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>CARLA Camera Views</title>
    <style>
      body { margin: 0; background: #000; color: #eee; font-family: sans-serif; }
      .row { display: flex; flex-direction: row; justify-content: space-around; align-items: flex-start; padding: 8px; }
      .panel { display: flex; flex-direction: column; align-items: center; }
      img { max-width: 100%; height: auto; border: 1px solid #444; }
      h2 { margin: 4px 0; font-size: 14px; }
      pre { background: #111; color: #0f0; padding: 8px; border: 1px solid #333; min-width: 200px; min-height: 120px; }
    </style>
  </head>
  <body>
    <div class="row">
      <div class="panel">
        <h2>Front Camera (Ego View)</h2>
        <img id="cam_front" src="/image_front" />
      </div>
      <div class="panel">
        <h2>Top-down Camera (Bird's-eye)</h2>
        <img id="cam_top" src="/image_top" />
      </div>
      <div class="panel">
        <h2>Speeds (km/h) & Acc (m/s^2)</h2>
        <pre id="speeds">Loading...</pre>
      </div>
      <div class="panel">
        <h2>VLM Output (latest)</h2>
        <pre id="vlm">Loading...</pre>
      </div>
    </div>
    <script>
      const imgFront = document.getElementById('cam_front');
      const imgTop = document.getElementById('cam_top');
      const speedsBox = document.getElementById('speeds');
      const vlmBox = document.getElementById('vlm');
      setInterval(() => {
        const t = Date.now();
        imgFront.src = '/image_front?t=' + t;
        imgTop.src = '/image_top?t=' + t;
        fetch('/speeds?t=' + t)
          .then(r => r.text())
          .then(txt => { speedsBox.textContent = txt || 'No data'; })
          .catch(() => { speedsBox.textContent = 'No data'; });
        fetch('/vlm?t=' + t)
          .then(r => r.text())
          .then(txt => { vlmBox.textContent = txt || 'No data'; })
          .catch(() => { vlmBox.textContent = 'No data'; });
      }, 100);
    </script>
  </body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path.startswith("/image_front"):
            with latest_lock:
                data = latest_front_jpeg
            if data is None:
                self.send_response(204)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/image_top"):
            with latest_lock:
                data = latest_top_jpeg
            if data is None:
                self.send_response(204)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/speeds"):
            with status_lock:
                data = latest_speed_text
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))
            return

        if self.path.startswith("/vlm"):
            with status_lock:
                data = latest_vlm_text
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data.encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def main(args: argparse.Namespace):
    client = carla.Client("localhost", 2000)
    # 加长超时时间，方便加载新地图
    client.set_timeout(30.0)

    # 选择地图（注意：不同 CARLA 安装/服务器可能不包含全部 TownXX）
    map_name = "Town04"
    try:
        world = client.load_world(map_name)
        print("Loaded map:", map_name)
    except RuntimeError:
        # 如服务器缺少该地图，则退回当前地图，仍可用直线坐标系近似
        world = client.get_world()
        print(
            f"Warning: failed to load {map_name}, using current map: {world.get_map().name}"
        )
        try:
            maps = client.get_available_maps()
            print("Available maps on server:", maps)
        except Exception as e:
            print("Failed to query available maps:", repr(e))
    original_settings = world.get_settings()

    av_vehicle = None
    hv_vehicles = []
    camera_front = None
    camera_top = None
    http_server: HTTPServer | None = None
    http_thread: threading.Thread | None = None
    http_port_in_use = int(getattr(args, "http_port", 8080))

    # ===== 录制 mp4：提前定义，避免异常路径下 finally 引用未赋值变量 =====
    record_enabled = bool(getattr(args, "record_mp4", 0))
    record_every_n = max(int(getattr(args, "record_every_n", 2)), 1)
    record_max_side = max(int(getattr(args, "record_max_side", 640)), 64)
    record_include_top = bool(getattr(args, "record_include_top", 1))
    record_writer = None
    record_path = None

    try:
        # 启动 HTTP 服务器，用浏览器可视化相机画面
        class _ReusableHTTPServer(HTTPServer):
            allow_reuse_address = True

        def _try_bind_http_server(
            start_port: int, max_tries: int = 20
        ) -> tuple[HTTPServer, int]:
            """
            绑定 HTTP 端口。如果端口被占用（常见于上次 ^Z 挂起或异常退出），
            则自动尝试后续端口：start_port, start_port+1, ...
            """
            last_err: OSError | None = None
            for p in range(start_port, start_port + max_tries):
                try:
                    srv = _ReusableHTTPServer(("0.0.0.0", int(p)), CameraHTTPHandler)
                    return srv, int(p)
                except OSError as e:
                    last_err = e
                    if getattr(e, "errno", None) == 98:
                        continue
                    raise
            raise OSError(
                f"Failed to bind HTTP server starting at port {start_port} "
                f"(tried {max_tries} ports). Last error: {last_err!r}"
            )

        http_server, http_port_in_use = _try_bind_http_server(int(args.http_port))
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        print(f"HTTP camera server running on http://localhost:{http_port_in_use}")

        # 切换为同步模式，固定时间步长
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 统一使用同一车型，减少干扰
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        # 清理当前世界中已有的车辆，避免与我们新场景发生碰撞
        for actor in world.get_actors().filter("vehicle.*"):
            actor.destroy()

        # 选取一个已有的可用出生点作为「场景原点」，再根据作业给定的
        # 纵向/横向偏移构造 1 AV + 5 HV 的三车道布局。
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in current map.")

        base_tf = spawn_points[BASE_SPAWN_INDEX % len(spawn_points)]
        base_loc = base_tf.location
        base_rot = base_tf.rotation
        forward = base_tf.get_forward_vector()
        right = base_tf.get_right_vector()
        forward_np = np.array([forward.x, forward.y, forward.z], dtype=float)
        right_np = np.array([right.x, right.y, right.z], dtype=float)
        forward_np /= np.linalg.norm(forward_np) + 1e-8
        right_np /= np.linalg.norm(right_np) + 1e-8

        def world_to_sd(loc: carla.Location) -> tuple[float, float]:
            """把世界坐标映射到以 base_tf 为原点的 (s, d)。"""
            rel = np.array(
                [loc.x - base_loc.x, loc.y - base_loc.y, loc.z - base_loc.z],
                dtype=float,
            )
            s_val = float(np.dot(rel, forward_np))
            d_val = float(np.dot(rel, right_np))
            return s_val, d_val

        def make_transform(longitudinal: float, lateral: float) -> carla.Transform:
            """
            longitudinal: 沿道路前进方向的偏移（m），正方向为前方。
            lateral: 沿车道横向（右手方向）的偏移（m），正为右、负为左。
            """
            lateral = lateral - LANE_SHIFT
            loc = carla.Location(
                x=base_loc.x + forward.x * longitudinal + right.x * lateral,
                y=base_loc.y + forward.y * longitudinal + right.y * lateral,
                z=base_loc.z + 0.3,
            )
            return carla.Transform(loc, base_rot)

        # AV：中间车道，定义为 (s=0, d=0)
        av_transform = make_transform(0.0, 0.0)
        av_vehicle = world.try_spawn_actor(vehicle_bp, av_transform)
        if av_vehicle is None:
            raise RuntimeError("Failed to spawn AV vehicle (collision at spawn position).")
        print("Spawned AV at", av_transform.location)
        # 设定 AV 初始速度（80 km/h）沿道路前进方向
        av_init_vec = carla.Vector3D(
            float(forward_np[0] * AV_INIT_SPEED),
            float(forward_np[1] * AV_INIT_SPEED),
            float(forward_np[2] * AV_INIT_SPEED),
        )
        av_vehicle.set_target_velocity(av_init_vec)

        # HV 列表：使用 IDM 控制纵向（车道固定）
        hv_list: list[IDMVehicle] = []

        def add_hv(
            name: str,
            actor: carla.Actor | None,
            s0: float,
            d0: float,
            speed_kmh: float,
            style: str,
        ):
            """注册一辆 HV：初始纵向/横向位置 + 初始速度（km/h）+ 驾驶风格。"""
            if actor is None:
                return
            v0 = speed_kmh / 3.6
            hv_list.append(
                IDMVehicle(
                    name=name,
                    actor=actor,
                    s0=s0,
                    d0=d0,
                    v0=v0,
                    style_params=IDM_STYLES[style],
                    v_desired=v0,  # 期望速度按该车初始设定速度
                )
            )

        # HV1：同车道前车，中间车道，前方 50 m，v = 70 km/h
        hv1 = world.try_spawn_actor(vehicle_bp, make_transform(50.0, 0.0))
        add_hv("HV1", hv1, 50.0, 0.0, 70.0, "normal")

        # HV2：左侧车道，AV 后方 20 m，v = 75 km/h
        hv2 = world.try_spawn_actor(vehicle_bp, make_transform(-20.0, +LANE_WIDTH))
        add_hv("HV2", hv2, -20.0, +LANE_WIDTH, 75.0, "conservative")

        # HV3：左侧车道，AV 前方 80 m，v = 85 km/h
        hv3 = world.try_spawn_actor(vehicle_bp, make_transform(80.0, +LANE_WIDTH))
        add_hv("HV3", hv3, 80.0, +LANE_WIDTH, 85.0, "aggressive")

        # HV4：右侧车道，AV 前方 10 m，v = 90 km/h
        hv4 = world.try_spawn_actor(vehicle_bp, make_transform(10.0, -LANE_WIDTH))
        add_hv("HV4", hv4, 10.0, -LANE_WIDTH, 90.0, "normal")

        # HV5：右侧车道，AV 后方 30 m，v = 75 km/h
        hv5 = world.try_spawn_actor(vehicle_bp, make_transform(-30.0, -LANE_WIDTH))
        add_hv("HV5", hv5, -30.0, -LANE_WIDTH, 75.0, "conservative")

        hv_vehicles = [hv.actor for hv in hv_list]
        print(f"Spawned {len(hv_vehicles)} HVs for three-lane scenario.")

        # ===== 创建并挂载两台 RGB 相机到 AV（第一人称 + 鸟瞰） =====
        camera_bp_front = blueprint_library.find("sensor.camera.rgb")
        camera_bp_top = blueprint_library.find("sensor.camera.rgb")

        image_w = 800
        image_h = 600
        for bp in (camera_bp_front, camera_bp_top):
            bp.set_attribute("image_size_x", str(image_w))
            bp.set_attribute("image_size_y", str(image_h))
            bp.set_attribute("fov", "90")

        # 第一人称相机：车头略微俯视
        cam_front_tf = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0),
        )
        # 鸟瞰相机：车体上方俯视
        cam_top_tf = carla.Transform(
            carla.Location(x=0.0, z=40.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

        front_queue: "queue.Queue[carla.Image]" = queue.Queue()
        top_queue: "queue.Queue[carla.Image]" = queue.Queue()

        camera_front = world.spawn_actor(
            camera_bp_front,
            cam_front_tf,
            attach_to=av_vehicle,
        )
        camera_front.listen(front_queue.put)

        camera_top = world.spawn_actor(
            camera_bp_top,
            cam_top_tf,
            attach_to=av_vehicle,
        )
        camera_top.listen(top_queue.put)

        # 让 AV 以简单控制向前开一小段时间（后续可替换为 IDM/FSM 控制）
        running = True
        frame_count = 0
        dt = settings.fixed_delta_seconds or 0.05
        sim_fps = float(1.0 / dt) if dt > 1e-6 else 20.0

        # ===== 录制 mp4（依赖 conda 环境中的 ffmpeg）=====
        if record_enabled:
            out_dir = Path(__file__).resolve().parents[2] / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_id = time.strftime("%Y%m%d-%H%M%S")
            record_path = out_dir / f"sim_{run_id}.mp4"

        # AV 控制模式（可通过命令行或环境变量配置）：
        # - cruise：简易巡航（默认）
        # - drivevla_replay：读取 DriveVLA 推理输出动作并回放（需要 DRIVEVLA_ACTIONS 路径）
        # - drivevla_live：OpenDriveVLA 在线推理（需要 OpenDriveVLA 依赖/权重）
        # - vlm_live：通用多模态大模型在线决策（OpenAI-compatible /v1/chat/completions）
        av_mode = args.av_mode
        drivevla_policy = None
        drivevla_pending_step = None
        vlm_policy = None
        vlm_pending_step = None
        if av_mode == "drivevla_replay":
            if DriveVLAReplayPolicy is None:
                raise RuntimeError("AV_MODE=drivevla_replay 但无法导入 drivevla_av.py（检查文件是否存在/依赖）")
            drivevla_actions = os.environ.get("DRIVEVLA_ACTIONS", "").strip()
            if not drivevla_actions:
                raise RuntimeError(
                    "AV_MODE=drivevla_replay 需要设置环境变量 DRIVEVLA_ACTIONS 指向 DriveVLA 输出 json 文件或目录"
                )
            # 默认取 horizon 第 0 个 waypoint
            drivevla_policy = DriveVLAReplayPolicy(drivevla_actions, horizon_index=0)
        elif av_mode == "drivevla_live":
            if DriveVLALivePolicy is None:
                raise RuntimeError("AV_MODE=drivevla_live 但无法导入 DriveVLALivePolicy（检查 OpenDriveVLA 依赖/路径）")
            # OpenDriveVLA 根目录（默认使用本仓库下的 ./OpenDriveVLA）
            default_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "OpenDriveVLA")
            )
            opendrivevla_root = os.environ.get("OPENDRIVEVLA_ROOT", default_root).strip()
            # checkpoint/模型路径：可以是本地路径，也可以是 HuggingFace repo id（如 OpenDriveVLA/OpenDriveVLA-0.5B）
            default_model = os.path.join(opendrivevla_root, "checkpoints", "DriveVLA-Qwen2.5-0.5B-Instruct")
            model_path = os.environ.get("OPENDRIVEVLA_MODEL_PATH", default_model).strip()
            min_interval_s = float(os.environ.get("OPENDRIVEVLA_MIN_INTERVAL_S", "0.5"))
            use_bf16 = os.environ.get("OPENDRIVEVLA_BF16", "1").strip() not in ("0", "false", "False")
            device = os.environ.get("OPENDRIVEVLA_DEVICE", "cuda").strip()
            drivevla_policy = DriveVLALivePolicy(
                opendrivevla_root=opendrivevla_root,
                model_path=model_path,
                device=device,
                use_bf16=use_bf16,
                min_interval_s=min_interval_s,
                # 默认开启“视觉+规划”：会构造最小 uniad_data 并使用 <scene>/<track>/<map> token 注入视觉特征
                use_images=os.environ.get("OPENDRIVEVLA_USE_VISION", "1").strip() not in ("0", "false", "False"),
            )
            print(drivevla_policy)
        elif av_mode == "vlm_live":
            if OpenAICompatibleVLMLivePolicy is None:
                raise RuntimeError("AV_MODE=vlm_live 但无法导入 vlm_av.py（检查 requests/Pillow 依赖）")
            # 参考 OpenEQA/infer.py：base_url 可按 model_name 自动选择；也可显式指定 VLM_API_BASE
            api_base = args.vlm_api_base
            api_key = args.vlm_api_key
            model = args.vlm_model
            timeout_s = args.vlm_timeout_s
            min_interval_s = args.vlm_min_interval_s
            max_image_side = args.vlm_max_image_side
            vlm_policy = OpenAICompatibleVLMLivePolicy(
                api_base=api_base,
                model=model,
                api_key=api_key,
                timeout_s=timeout_s,
                min_interval_s=min_interval_s,
                max_image_side=max_image_side,
                memory_steps=int(args.vlm_memory_steps),
            )
            print(vlm_policy)

        def get_av_state() -> KinematicState | None:
            if av_vehicle is None or not av_vehicle.is_alive:
                return None
            tf = av_vehicle.get_transform()
            s_val, d_val = world_to_sd(tf.location)
            # d_val 是 world-d，make_transform 的入参是 shifted-d，需要加上 LANE_SHIFT 做对齐
            d_shifted = d_val + LANE_SHIFT
            vel = av_vehicle.get_velocity()
            v_long = float(
                vel.x * forward_np[0]
                + vel.y * forward_np[1]
                + vel.z * forward_np[2]
            )
            return KinematicState(
                s=s_val,
                v=max(v_long, 0.0),
                d=d_shifted,
                lane=lane_index_from_d(d_shifted),
            )

        def log_vehicle_speeds(
            av_state: KinematicState | None,
            av_acc: float,
            frame_idx: int,
            action_step=None,
        ) -> None:
            entries: list[str] = []
            if av_state:
                entries.append(f"AV:{av_state.v * 3.6:.2f}(a:{av_acc:.2f})")
            else:
                entries.append("AV:NA(a:NA)")
            for hv in hv_list:
                entries.append(f"{hv.name}:{hv.v * 3.6:.2f}(a:{hv.a:.2f})")
            # 在状态面板上显示最新动作（如果有）
            if action_step is not None:
                try:
                    entries.append(
                        f"Action(dx,dy,dyaw): {action_step.dx:+.2f}, {action_step.dy:+.2f}, {action_step.dyaw:+.3f}"
                    )
                except Exception:
                    # 容错：action_step 不是预期类型时不影响主循环
                    pass

            text = "Speeds/Acc (km/h, m/s^2)\n" + "\n".join(entries)
            with status_lock:
                global latest_speed_text
                latest_speed_text = text

        # 初始化一次速度面板文本，避免网页初期为空
        av_prev_v = 0.0
        av_state_init = get_av_state()
        if av_state_init:
            av_prev_v = av_state_init.v
        log_vehicle_speeds(av_state_init, 0.0, frame_count, action_step=None)

        latest_pil_front: Image.Image | None = None
        latest_pil_top: Image.Image | None = None
        last_action_print_t = 0.0
        print_action = bool(args.print_av_action)
        print_action_interval_s = float(os.environ.get("PRINT_AV_ACTION_INTERVAL_S", "0.5"))

        while running and frame_count < 400:  # 跑约 20 秒
            frame_count += 1

            av_state = get_av_state()
            av_acc = 0.0
            if av_state:
                av_acc = (av_state.v - av_prev_v) / dt
                av_prev_v = av_state.v

            if av_mode == "drivevla_replay" and drivevla_policy is not None:
                # 通过 set_transform 以“运动学回放”方式执行（最稳妥，先把链路打通）
                step = drivevla_policy.step()
                if step is not None and av_state is not None:
                    # 将 DriveVLA 的相对 (dx,dy) 映射到本项目的 (s,d)
                    new_s = av_state.s + step.dx
                    new_d = av_state.d + step.dy
                    av_vehicle.set_transform(make_transform(new_s, new_d))
                    if print_action and (time.time() - last_action_print_t) >= print_action_interval_s:
                        last_action_print_t = time.time()
                        print(
                            f"[frame {frame_count:03d}] AV action(replay): dx={step.dx:+.2f} dy={step.dy:+.2f} dyaw={step.dyaw:+.3f}"
                        )
                else:
                    # 回放结束或无数据：保持不动
                    pass
            elif av_mode == "drivevla_live" and drivevla_policy is not None:
                # 使用上一轮推理得到的动作（避免在同步 tick 内阻塞太久）
                if drivevla_pending_step is not None and av_state is not None:
                    new_s = av_state.s + drivevla_pending_step.dx
                    new_d = av_state.d + drivevla_pending_step.dy
                    av_vehicle.set_transform(make_transform(new_s, new_d))
            elif av_mode == "vlm_live" and vlm_policy is not None:
                # 使用上一轮 VLM 决策得到的动作（避免在同步 tick 内阻塞太久）
                if vlm_pending_step is not None and av_state is not None:
                    # 再做一层安全限幅：每步最多前进 3m、横向 1.5m
                    dx = float(np.clip(vlm_pending_step.dx, 0.0, 3.0))
                    dy = float(np.clip(vlm_pending_step.dy, -1.5, 1.5))
                    new_s = av_state.s + dx
                    new_d = av_state.d + dy
                    av_vehicle.set_transform(make_transform(new_s, new_d))
                else:
                    # VLM 暂无输出：退化为巡航（避免车辆“卡住”）
                    cur_v = av_state.v if av_state else 0.0
                    err_v = V_DESIRED - cur_v
                    throttle_cmd = np.clip(0.1 * err_v, 0.0, 0.7)
                    brake_cmd = np.clip(-0.2 * err_v, 0.0, 1.0)
                    control = carla.VehicleControl(
                        throttle=float(throttle_cmd), brake=float(brake_cmd), steer=0.0
                    )
                    av_vehicle.apply_control(control)
            else:
                # AV 简易巡航控制：保持期望速度 V_DESIRED
                cur_v = av_state.v if av_state else 0.0
                err_v = V_DESIRED - cur_v
                throttle_cmd = np.clip(0.1 * err_v, 0.0, 0.7)
                brake_cmd = np.clip(-0.2 * err_v, 0.0, 1.0)
                control = carla.VehicleControl(
                    throttle=float(throttle_cmd), brake=float(brake_cmd), steer=0.0
                )
                av_vehicle.apply_control(control)

            # HV：IDM 纵向控制（车道固定）
            for hv in hv_list:
                if hv.actor is None or not hv.actor.is_alive:
                    continue

                front_candidates: list[KinematicState | IDMVehicle] = []
                if av_state and av_state.lane == hv.lane and av_state.s > hv.s:
                    front_candidates.append(av_state)

                for other in hv_list:
                    if other is hv:
                        continue
                    if other.lane == hv.lane and other.s > hv.s:
                        front_candidates.append(other)

                front_vehicle = (
                    min(front_candidates, key=lambda obj: obj.s)
                    if front_candidates
                    else None
                )
                hv.step(dt, front_vehicle, make_transform)

            # UI 上显示动作：优先显示 live 的 pending_step，其次显示 replay step（在 replay 分支已直接打印）
            log_vehicle_speeds(
                av_state,
                av_acc,
                frame_count,
                action_step=(drivevla_pending_step if av_mode == "drivevla_live" else None),
            )

            # 同步步进
            world.tick()

            # 从两个相机队列中取最近一帧图像并更新 HTTP 流
            try:
                img_front = front_queue.get(timeout=0.05)
                arr_f = np.frombuffer(img_front.raw_data, dtype=np.uint8)
                arr_f = arr_f.reshape((img_front.height, img_front.width, 4))
                arr_f = arr_f[:, :, :3][:, :, ::-1]  # BGRA -> RGB
                pil_f = Image.fromarray(arr_f)
                latest_pil_front = pil_f
                buf_f = io.BytesIO()
                pil_f.save(buf_f, format="JPEG", quality=80)
                with latest_lock:
                    global latest_front_jpeg
                    latest_front_jpeg = buf_f.getvalue()
            except queue.Empty:
                pass

            try:
                img_top = top_queue.get(timeout=0.05)
                arr_t = np.frombuffer(img_top.raw_data, dtype=np.uint8)
                arr_t = arr_t.reshape((img_top.height, img_top.width, 4))
                arr_t = arr_t[:, :, :3][:, :, ::-1]  # BGRA -> RGB
                pil_t = Image.fromarray(arr_t)
                latest_pil_top = pil_t
                buf_t = io.BytesIO()
                pil_t.save(buf_t, format="JPEG", quality=80)
                with latest_lock:
                    global latest_top_jpeg
                    latest_top_jpeg = buf_t.getvalue()
            except queue.Empty:
                pass

            # ===== 写入 mp4：合并前视 + 鸟瞰（可选）=====
            if (
                record_enabled
                and record_path is not None
                and (frame_count % record_every_n == 0)
                and latest_pil_front is not None
            ):
                try:
                    # 延迟导入，避免非录制模式额外依赖
                    import imageio.v2 as imageio  # type: ignore

                    f_img = _resize_for_record(latest_pil_front.convert("RGB"), record_max_side)
                    if record_include_top and latest_pil_top is not None:
                        t_img = _resize_for_record(latest_pil_top.convert("RGB"), record_max_side)
                        # 对齐高度后水平拼接
                        if t_img.size[1] != f_img.size[1]:
                            t_img = t_img.resize((t_img.size[0], f_img.size[1]), Image.BICUBIC)
                        merged = Image.new("RGB", (f_img.size[0] + t_img.size[0], f_img.size[1]))
                        merged.paste(f_img, (0, 0))
                        merged.paste(t_img, (f_img.size[0], 0))
                    else:
                        merged = f_img

                    merged = _pad_to_multiple(merged, 16)
                    if record_writer is None:
                        fps_out = sim_fps / float(record_every_n)
                        record_writer = imageio.get_writer(
                            str(record_path),
                            fps=fps_out,
                            codec="libx264",
                            quality=8,
                        )
                    record_writer.append_data(np.asarray(merged))
                except Exception as e:
                    print("[record] mp4 write failed:", repr(e))
                    record_enabled = False  # 避免刷屏

            # drivevla_live：在拿到最新相机帧后做一次推理，供下一轮使用
            if av_mode == "drivevla_live" and drivevla_policy is not None:
                try:
                    av_state_now = get_av_state()
                    if av_state_now is not None and latest_pil_front is not None:
                        drivevla_pending_step = drivevla_policy.step(
                            front_pil_image=latest_pil_front,
                            top_pil_image=latest_pil_top,
                            speed_mps=av_state_now.v,
                        )
                        if (
                            print_action
                            and drivevla_pending_step is not None
                            and (time.time() - last_action_print_t) >= print_action_interval_s
                        ):
                            last_action_print_t = time.time()
                            print(
                                f"[frame {frame_count:03d}] AV action(live): dx={drivevla_pending_step.dx:+.2f} "
                                f"dy={drivevla_pending_step.dy:+.2f} dyaw={drivevla_pending_step.dyaw:+.3f}"
                            )
                except Exception as e:
                    # 在线推理失败时退化为“保持上一次动作/不动作”
                    drivevla_pending_step = None
                    print("DriveVLA live inference failed:", repr(e))

            # vlm_live：在拿到最新相机帧后做一次决策，供下一轮使用
            if av_mode == "vlm_live" and vlm_policy is not None:
                try:
                    av_state_now = get_av_state()
                    if av_state_now is not None and latest_pil_front is not None:
                        use_top = bool(args.vlm_use_top)
                        vlm_pending_step = vlm_policy.step(
                            front_pil_image=latest_pil_front,
                            top_pil_image=(latest_pil_top if use_top else None),
                            speed_mps=av_state_now.v,
                            lane=av_state_now.lane,
                        )
                        # 把模型返回内容写入网页面板（/vlm）
                        if vlm_pending_step is not None:
                            with status_lock:
                                global latest_vlm_text
                                latest_vlm_text = json.dumps(
                                    {
                                        "dx": float(vlm_pending_step.dx),
                                        "dy": float(vlm_pending_step.dy),
                                        "dyaw": float(vlm_pending_step.dyaw),
                                        "reason": str(getattr(vlm_pending_step, "reason", "") or ""),
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        if (
                            print_action
                            and vlm_pending_step is not None
                            and (time.time() - last_action_print_t) >= print_action_interval_s
                        ):
                            last_action_print_t = time.time()
                            print(
                                f"[frame {frame_count:03d}] AV action(vlm): dx={vlm_pending_step.dx:+.2f} "
                                f"dy={vlm_pending_step.dy:+.2f} dyaw={vlm_pending_step.dyaw:+.3f}"
                            )
                except Exception as e:
                    vlm_pending_step = None
                    print("VLM live inference failed:", repr(e))

        print("*"*20,"仿真结束","*"*20)

    finally:
        # 关闭录制器并打印输出路径
        try:
            if record_writer is not None:
                record_writer.close()
                print(f"[record] saved mp4: {record_path}")
        except Exception as e:
            print("[record] finalize failed:", repr(e))

        # 停止相机并销毁 actor，恢复仿真设置
        if camera_front is not None:
            camera_front.stop()
            camera_front.destroy()

        if camera_top is not None:
            camera_top.stop()
            camera_top.destroy()

        if av_vehicle is not None and av_vehicle.is_alive:
            av_vehicle.destroy()

        for hv in hv_vehicles:
            if hv is not None and hv.is_alive:
                hv.destroy()

        if http_server is not None:
            try:
                http_server.shutdown()
            finally:
                # 彻底释放端口；仅 shutdown 可能仍让端口处于占用/WAIT 状态
                http_server.server_close()
        if http_thread is not None and http_thread.is_alive():
            http_thread.join(timeout=1.0)

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        print("Cleaned up actors, restored settings, and stopped HTTP server.")


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)


