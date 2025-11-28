import carla
import json
import queue
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import io

import numpy as np
from PIL import Image


# 最新一帧相机图像（JPEG 编码）：前向视角 + 鸟瞰视角
latest_front_jpeg: bytes | None = None
latest_top_jpeg: bytes | None = None
latest_lock = threading.Lock()

# 来自键盘的控制状态（WASD + 空格）
control_state = {
    "w": False,  # 前进（油门）
    "s": False,  # 刹车 / 倒退
    "a": False,  # 左转
    "d": False,  # 右转
    "space": False,  # 刹车
}
control_lock = threading.Lock()


class KeyboardCameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_index()
        elif self.path.startswith("/image_front"):
            self._serve_image_front()
        elif self.path.startswith("/image_top"):
            self._serve_image_top()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/control":
            self._handle_control()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_index(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>CARLA Keyboard Control</title>
    <style>
      body { margin: 0; background: #000; color: #eee; font-family: sans-serif; }
      #info { padding: 8px 16px; text-align: center; }
      .keys { margin-top: 8px; }
      .key { display: inline-block; padding: 4px 8px; border: 1px solid #666; margin: 0 4px; border-radius: 4px; }
      .active { background: #3a6ff7; border-color: #9ab3ff; }
      .row { display: flex; flex-direction: row; justify-content: space-around; align-items: flex-start; padding: 8px; }
      .panel { display: flex; flex-direction: column; align-items: center; }
      img { max-width: 100%; height: auto; border: 1px solid #444; display: block; }
      h2 { margin: 4px 0; font-size: 14px; }
      .keys { margin-top: 8px; }
    </style>
  </head>
  <body>
    <div id="info">
      <div>使用键盘 <b>W/A/S/D</b> 控制车辆，<b>空格</b> 刹车。</div>
      <div class="keys">
        <span id="key-w" class="key">W</span>
        <span id="key-a" class="key">A</span>
        <span id="key-s" class="key">S</span>
        <span id="key-d" class="key">D</span>
        <span id="key-space" class="key">SPACE</span>
      </div>
    </div>
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
      const camFront = document.getElementById('cam_front');
      const camTop = document.getElementById('cam_top');
      const state = { w: false, a: false, s: false, d: false, space: false };

      const keyMap = {
        'w': 'w',
        'a': 'a',
        's': 's',
        'd': 'd',
        ' ': 'space',
        'arrowup': 'w',
        'arrowleft': 'a',
        'arrowdown': 's',
        'arrowright': 'd',
      };

      function updateKeys() {
        for (const [k, v] of Object.entries(state)) {
          const el = document.getElementById('key-' + (k === 'space' ? 'space' : k));
          if (!el) continue;
          if (v) el.classList.add('active');
          else el.classList.remove('active');
        }
      }

      async function sendState() {
        try {
          await fetch('/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state),
          });
        } catch (e) {
          console.error('sendState error', e);
        }
      }

      document.addEventListener('keydown', (e) => {
        const key = keyMap[e.key.toLowerCase()];
        if (!key) return;
        if (!state[key]) {
          state[key] = true;
          updateKeys();
          sendState();
        }
        e.preventDefault();
      });

      document.addEventListener('keyup', (e) => {
        const key = keyMap[e.key.toLowerCase()];
        if (!key) return;
        if (state[key]) {
          state[key] = false;
          updateKeys();
          sendState();
        }
        e.preventDefault();
      });

      // 定时刷新相机画面
      setInterval(() => {
        const t = Date.now();
        camFront.src = '/image_front?t=' + t;
        camTop.src = '/image_top?t=' + t;
      }, 100);
    </script>
  </body>
</html>
"""
        self.wfile.write(html.encode("utf-8"))

    def _serve_image_front(self):
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

    def _serve_image_top(self):
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

    def _handle_control(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        with control_lock:
            for key in control_state.keys():
                control_state[key] = bool(payload.get(key, False))

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    original_settings = world.get_settings()

    vehicle = None
    camera_front = None
    camera_top = None
    http_server = None
    http_thread = None

    try:
        # 启动 HTTP 服务器：键盘控制 + 相机画面
        http_server = HTTPServer(("0.0.0.0", 8080), KeyboardCameraHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        print("Keyboard + camera server on http://localhost:8080")

        # 同步模式
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 车辆
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Spawned vehicle:", vehicle.type_id, "at", spawn_point.location)

        # RGB 相机：前向视角 + 鸟瞰视角
        camera_bp_front = blueprint_library.find("sensor.camera.rgb")
        camera_bp_top = blueprint_library.find("sensor.camera.rgb")
        image_w, image_h = 800, 600
        for bp in (camera_bp_front, camera_bp_top):
            bp.set_attribute("image_size_x", str(image_w))
            bp.set_attribute("image_size_y", str(image_h))
            bp.set_attribute("fov", "90")

        cam_front_tf = carla.Transform(
            carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10.0)
        )
        cam_top_tf = carla.Transform(
            carla.Location(x=0.0, z=40.0), carla.Rotation(pitch=-90.0)
        )

        front_queue: "queue.Queue[carla.Image]" = queue.Queue()
        top_queue: "queue.Queue[carla.Image]" = queue.Queue()

        camera_front = world.spawn_actor(
            camera_bp_front,
            cam_front_tf,
            attach_to=vehicle,
        )
        camera_front.listen(front_queue.put)

        camera_top = world.spawn_actor(
            camera_bp_top,
            cam_top_tf,
            attach_to=vehicle,
        )
        camera_top.listen(top_queue.put)

        # 关闭 autopilot，只用我们自己的控制
        vehicle.set_autopilot(False)

        print("Use W/A/S/D + SPACE in the browser page to drive the vehicle.")

        running = True
        frame_count = 0

        while running and frame_count < 2000:  # 大约 100 秒
            frame_count += 1

            # 根据当前键盘状态生成控制量
            with control_lock:
                st = dict(control_state)

            throttle = 0.0
            brake = 0.0
            steer = 0.0
            reverse = False

            # 前进：W
            if st["w"]:
                throttle = 0.6

            # 后退 / 刹车：S
            if st["s"]:
                if st["w"]:
                    # 同时按 W+S，则视为刹车
                    brake = max(brake, 0.7)
                else:
                    # 只按 S，则倒车
                    reverse = True
                    throttle = 0.5

            # 空格：强力刹车
            if st["space"]:
                brake = 1.0
            if st["a"]:
                steer -= 0.5
            if st["d"]:
                steer += 0.5

            steer = max(-1.0, min(1.0, steer))

            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                reverse=reverse,
            )
            vehicle.apply_control(control)

            # 同步步进
            world.tick()

            # 相机图像：更新前向与鸟瞰两路 HTTP 流
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

        print("Keyboard control session finished.")

    finally:
        if camera_front is not None:
            camera_front.stop()
            camera_front.destroy()

        if camera_top is not None:
            camera_top.stop()
            camera_top.destroy()

        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()

        if http_server is not None:
            http_server.shutdown()

        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        print("Cleaned up actors, restored settings, and stopped HTTP server.")


if __name__ == "__main__":
    main()


