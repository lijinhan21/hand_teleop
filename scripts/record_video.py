import pyrealsense2 as rs
import numpy as np
import cv2
import time
from pathlib import Path

class RGBDRecorder:

    def __init__(self, output_path="recordings"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        # 配置RealSense流
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 启用彩色和深度流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动录制
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.rgb_video_path = str(self.output_path / f"rgb_{self.timestamp}.avi")
        self.depth_video_path = str(self.output_path / f"depth_{self.timestamp}.avi")

    def start_recording(self, duration=10):

        # 启动相机流
        self.pipeline.start(self.config)
        
        # 创建视频写入器
        rgb_writer = cv2.VideoWriter(
            self.rgb_video_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            30,
            (640, 480)
        )

        depth_writer = cv2.VideoWriter(
            self.depth_video_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            30,
            (640, 480)
        )

        start_time = time.time()
        try:
            while (time.time() - start_time) < duration:
                # 等待新的帧
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # 将深度图转换为可视化格式
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # 保存帧
                rgb_writer.write(color_image)
                depth_writer.write(depth_colormap)
                
                # 显示预览
                cv2.imshow('RGB-D Recording', np.hstack((color_image, depth_colormap)))
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:

            # 清理资源

            self.pipeline.stop()

            rgb_writer.release()

            depth_writer.release()

            cv2.destroyAllWindows()

            print(f"录制完成！\nRGB视频保存至: {self.rgb_video_path}\n深度视频保存至: {self.depth_video_path}")



if __name__ == "__main__":

    # 创建录制器实例并开始录制（默认10秒）

    recorder = RGBDRecorder()

    recorder.start_recording()