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
latest_lock = threading.Lock()

# 车道/场景参数（作业要求）
LANE_WIDTH = 3.5
LANE_RIGHT_Y = -LANE_WIDTH  # 右车道中心 y = -4.5
LANE_MID_Y = 0.0           # 中车道中心 y = 0
LANE_LEFT_Y = LANE_WIDTH   # 左车道中心 y = +4.5

# 车辆长度近似值（用于后续 IDM 等，先占位）
VEH_LENGTH = 3.5

# 选择用于构建直线路段的基础出生点索引（可根据需要调整）
BASE_SPAWN_INDEX = 100


class CameraHTTPHandler(BaseHTTPRequestHandler):
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
    </div>
    <script>
      const imgFront = document.getElementById('cam_front');
      const imgTop = document.getElementById('cam_top');
      setInterval(() => {
        const t = Date.now();
        imgFront.src = '/image_front?t=' + t;
        imgTop.src = '/image_top?t=' + t;
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

        self.send_response(404)
        self.end_headers()


def main():
    client = carla.Client("localhost", 2000)
    # 加长超时时间，方便加载新地图
    client.set_timeout(30.0)

    # 选择地图（注意：不同 CARLA 安装/服务器可能不包含全部 TownXX）
    map_name = "Town03"
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

        def make_transform(longitudinal: float, lateral: float) -> carla.Transform:
            """
            longitudinal: 沿道路前进方向的偏移（m），正方向为前方。
            lateral: 沿车道横向（右手方向）的偏移（m），正为右、负为左。
            """
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

        # HV 初始状态列表：用于后续简化纵向运动（常速前进，占位 IDM）
        hv_states: list[dict] = []

        def add_hv(
            name: str,
            actor: carla.Actor | None,
            s0: float,
            d0: float,
            speed_kmh: float,
        ):
            """注册一辆 HV：初始纵向位置 s0、横向位置 d0、初始速度（km/h）。"""
            if actor is None:
                return
            hv_states.append(
                {
                    "name": name,
                    "actor": actor,
                    "s": s0,
                    "d": d0,
                    "v": speed_kmh / 3.6,  # 转成 m/s
                }
            )

        # HV1：同车道前车，中间车道，前方 50 m，v = 70 km/h
        hv1 = world.try_spawn_actor(vehicle_bp, make_transform(50.0, 0.0))
        add_hv("HV1", hv1, 50.0, 0.0, 70.0)

        # HV2：左侧车道，AV 后方 20 m，v = 75 km/h
        hv2 = world.try_spawn_actor(vehicle_bp, make_transform(-20.0, +LANE_WIDTH))
        add_hv("HV2", hv2, -20.0, +LANE_WIDTH, 75.0)

        # HV3：左侧车道，AV 前方 80 m，v = 85 km/h
        hv3 = world.try_spawn_actor(vehicle_bp, make_transform(80.0, +LANE_WIDTH))
        add_hv("HV3", hv3, 80.0, +LANE_WIDTH, 85.0)

        # HV4：右侧车道，AV 前方 10 m，v = 90 km/h
        hv4 = world.try_spawn_actor(vehicle_bp, make_transform(10.0, -LANE_WIDTH))
        add_hv("HV4", hv4, 10.0, -LANE_WIDTH, 90.0)

        # HV5：右侧车道，AV 后方 30 m，v = 75 km/h
        hv5 = world.try_spawn_actor(vehicle_bp, make_transform(-30.0, -LANE_WIDTH))
        add_hv("HV5", hv5, -30.0, -LANE_WIDTH, 75.0)

        hv_vehicles = [s["actor"] for s in hv_states]
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

        while running and frame_count < 400:  # 跑约 20 秒
            frame_count += 1

            control = carla.VehicleControl(throttle=5, steer=0.0)
            av_vehicle.apply_control(control)

            # HV：先用简单「匀速直行」占位，将来替换为 IDM 更新
            for state in hv_states:
                actor = state["actor"]
                state["s"] += state["v"] * dt
                tf = make_transform(state["s"], state["d"])
                actor.set_transform(tf)

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


