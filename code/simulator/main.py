from __future__ import annotations

import carla
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

import numpy as np
from PIL import Image


# HTTP 相机最新一帧 JPEG（前向视角 & 鸟瞰视角）
latest_front_jpeg = None
latest_top_jpeg = None
latest_speed_text = "No data yet"
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
AV_INIT_SPEED = 80 / 3.6

# 三种驾驶风格参数
IDM_STYLES = {
    "conservative": {"a_max": 2.0, "b": 2.0, "s0": 5.0, "T": 2.0},
    "normal": {"a_max": 4.0, "b": 2.5, "s0": 3.0, "T": 1.5},
    "aggressive": {"a_max": 6.0, "b": 3.0, "s0": 2.0, "T": 1.0},
}


def lane_index_from_d(d: float) -> int:
    """根据横向偏移近似车道索引：左=+1，中=0，右=-1。"""
    return int(round((d + LANE_SHIFT) / LANE_WIDTH))


class KinematicState:
    """只存储纵向位置/速度/车道，用于前车搜索。"""

    __slots__ = ("s", "v", "lane")

    def __init__(self, s: float, v: float, lane: int):
        self.s = s
        self.v = v
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
        self.d = d0
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
    </div>
    <script>
      const imgFront = document.getElementById('cam_front');
      const imgTop = document.getElementById('cam_top');
      const speedsBox = document.getElementById('speeds');
      setInterval(() => {
        const t = Date.now();
        imgFront.src = '/image_front?t=' + t;
        imgTop.src = '/image_top?t=' + t;
        fetch('/speeds?t=' + t)
          .then(r => r.text())
          .then(txt => { speedsBox.textContent = txt || 'No data'; })
          .catch(() => { speedsBox.textContent = 'No data'; });
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

        self.send_response(404)
        self.end_headers()


def main():
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

    try:
        # 启动 HTTP 服务器，用浏览器可视化相机画面
        http_server = HTTPServer(("0.0.0.0", 8080), CameraHTTPHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        print("HTTP camera server running on http://localhost:8080")

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

        def get_av_state() -> KinematicState | None:
            if av_vehicle is None or not av_vehicle.is_alive:
                return None
            tf = av_vehicle.get_transform()
            s_val, d_val = world_to_sd(tf.location)
            vel = av_vehicle.get_velocity()
            v_long = float(
                vel.x * forward_np[0]
                + vel.y * forward_np[1]
                + vel.z * forward_np[2]
            )
            return KinematicState(
                s=s_val,
                v=max(v_long, 0.0),
                lane=lane_index_from_d(d_val),
            )

        def log_vehicle_speeds(
            av_state: KinematicState | None, av_acc: float, frame_idx: int
        ) -> None:
            entries: list[str] = []
            if av_state:
                entries.append(f"AV:{av_state.v * 3.6:.2f}(a:{av_acc:.2f})")
            else:
                entries.append("AV:NA(a:NA)")
            for hv in hv_list:
                entries.append(f"{hv.name}:{hv.v * 3.6:.2f}(a:{hv.a:.2f})")
            line = " | ".join(entries)
            print(f"[frame {frame_idx:03d}] speeds(km/h): " + line)
            text = "Speeds/Acc (km/h, m/s^2)\n" + "\n".join(entries)
            with status_lock:
                global latest_speed_text
                latest_speed_text = text

        # 初始化一次速度面板文本，避免网页初期为空
        av_prev_v = 0.0
        av_state_init = get_av_state()
        if av_state_init:
            av_prev_v = av_state_init.v
        log_vehicle_speeds(av_state_init, 0.0, frame_count)

        while running and frame_count < 400:  # 跑约 20 秒
            frame_count += 1

            av_state = get_av_state()
            av_acc = 0.0
            if av_state:
                av_acc = (av_state.v - av_prev_v) / dt
                av_prev_v = av_state.v

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

            log_vehicle_speeds(av_state, av_acc, frame_count)

            # 同步步进
            world.tick()

            # 从两个相机队列中取最近一帧图像并更新 HTTP 流
            try:
                img_front = front_queue.get(timeout=0.05)
                arr_f = np.frombuffer(img_front.raw_data, dtype=np.uint8)
                arr_f = arr_f.reshape((img_front.height, img_front.width, 4))
                arr_f = arr_f[:, :, :3][:, :, ::-1]  # BGRA -> RGB
                pil_f = Image.fromarray(arr_f)
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
                buf_t = io.BytesIO()
                pil_t.save(buf_t, format="JPEG", quality=80)
                with latest_lock:
                    global latest_top_jpeg
                    latest_top_jpeg = buf_t.getvalue()
            except queue.Empty:
                pass

        print("Simulation with camera visualization finished.")

    finally:
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
            http_server.shutdown()

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        print("Cleaned up actors, restored settings, and stopped HTTP server.")


if __name__ == "__main__":
    main()


