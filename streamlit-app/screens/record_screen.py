# screens/record_screen.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from components.navigation import render_navbar, create_back_button
import av
import cv2
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip
import logging
from pathlib import Path
from typing import List
from scipy import interpolate
import threading
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
FPS = 30.0
OUTPUT_DIR = Path(__file__).parent.parent / 'temp'
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# Global state
frames = None
recording_state = {"is_recording": False}

class FrameSync:
    def __init__(self):
        self.audio_frames = []
        self.video_frames = []
        self._audio_lock = threading.Lock()
        self._video_lock = threading.Lock()
        self._start_time = None

    def add_audio_frames(self, frames_list):
        with self._audio_lock:
            for frame in frames_list:
                audio_data = frame.to_ndarray()
                timestamp = frame.time
                if self._start_time is None:
                    self._start_time = timestamp
                if len(audio_data.shape) == 2 and audio_data.shape[0] == 2:
                    audio_data = audio_data.mean(axis=0)
                self.audio_frames.append((audio_data, timestamp))

    def add_video_frame(self, frame, timestamp):
        with self._video_lock:
            if self._start_time is None:
                self._start_time = timestamp
            self.video_frames.append((frame, timestamp))

    def clear(self):
        with self._audio_lock:
            self.audio_frames.clear()
        with self._video_lock:
            self.video_frames.clear()
        self._start_time = None

class VideoProcessor:
    def __init__(self):
        global frames
        self.frames = frames

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if recording_state["is_recording"]:
            self.frames.add_video_frame(img.copy(), frame.time)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class AudioProcessor:
    def __init__(self):
        global frames
        self.frames = frames

    async def recv_queued(self, frames_list: List[av.AudioFrame]) -> List[av.AudioFrame]:
        if recording_state["is_recording"]:
            self.frames.add_audio_frames(frames_list)
        return frames_list

def save_recording(frame_sync):
    shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    """Save the recorded video and audio."""
    try:
        logger.info("Starting save process")
        with frame_sync._audio_lock, frame_sync._video_lock:
            if not frame_sync.audio_frames or not frame_sync.video_frames:
                st.warning("No recorded frames found.")
                return None

            sorted_audio_frames = sorted(frame_sync.audio_frames, key=lambda x: x[1])
            sorted_video_frames = sorted(frame_sync.video_frames, key=lambda x: x[1])

            logger.info(f"Total frames - Audio: {len(sorted_audio_frames)}, Video: {len(sorted_video_frames)}")

            video_start = sorted_video_frames[0][1]
            video_end = sorted_video_frames[-1][1]
            duration = video_end - video_start

            processed_video = [frame for frame, _ in sorted_video_frames]
            
            # Process audio frames
            processed_audio = []
            for frame, ts in sorted_audio_frames:
                if video_start <= ts <= video_end:
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 2:
                        frame = frame.mean(axis=0)
                    processed_audio.append(frame)

            if not processed_audio:
                st.warning("No synchronized audio frames found.")
                return None

            # Create output paths
            paths = {
                "video": OUTPUT_DIR / f'video.mp4',
                "audio": OUTPUT_DIR / f'audio.wav',
                "final": OUTPUT_DIR / f'final_video.mp4'
            }

            # Save video
            logger.info("Saving video frames...")
            try:
                height, width = processed_video[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                temp_avi = OUTPUT_DIR / f'temp_video.avi'
                out = cv2.VideoWriter(str(temp_avi), fourcc, FPS, (width, height))
                
                if not out.isOpened():
                    raise Exception("Failed to open video writer")
                
                for frame in processed_video:
                    out.write(frame)
                out.release()
                
                if temp_avi.exists():
                    video_clip = VideoFileClip(str(temp_avi))
                    video_clip.write_videofile(
                        str(paths["video"]),
                        codec='libx264',
                        preset='medium',
                        remove_temp=True,
                        logger=None
                    )
                    video_clip.close()
                    temp_avi.unlink(missing_ok=True)
                
            except Exception as e:
                logger.error(f"Error saving video: {str(e)}")
                raise

            # Process and save audio
            logger.info("Processing audio frames...")
            try:
                audio_data = np.concatenate(processed_audio)
                audio_data = audio_data.astype(np.float32)
                
                max_val = np.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data / max_val

                target_samples = int(duration * SAMPLE_RATE)
                current_samples = len(audio_data)

                if current_samples != target_samples and current_samples > 0:
                    current_time = np.linspace(0, duration, current_samples)
                    target_time = np.linspace(0, duration, target_samples)
                    f = interpolate.interp1d(current_time, audio_data, 
                                           bounds_error=False, fill_value="extrapolate")
                    audio_data = f(target_time)

                logger.info(f"Saving audio file to {paths['audio']}")
                sf.write(str(paths["audio"]), audio_data, SAMPLE_RATE, format='WAV', subtype='FLOAT')
                
                if not paths["audio"].exists():
                    raise Exception("Audio file was not created")
                logger.info("Audio file saved successfully")
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                raise

            # Combine audio and video
            logger.info("Combining audio and video...")
            try:
                if not paths["video"].exists():
                    raise Exception("Video file does not exist")
                if not paths["audio"].exists():
                    raise Exception("Audio file does not exist")
                
                logger.info("Loading video clip...")
                video_clip = VideoFileClip(str(paths["video"]))
                logger.info("Loading audio clip...")
                audio_clip = AudioFileClip(str(paths["audio"]))
                
                # 오디오 길이에 맞춰 비디오 속도 조정
                logger.info(f"Original - Video duration: {video_clip.duration}, Audio duration: {audio_clip.duration}")
                if abs(video_clip.duration - audio_clip.duration) > 0.1:  # 0.1초 이상 차이나면 조정
                    speed_factor = video_clip.duration / audio_clip.duration
                    video_clip = video_clip.speedx(speed_factor)
                    logger.info(f"Adjusted - Video duration: {video_clip.duration} (speed factor: {speed_factor:.2f}x)")
                
                logger.info("Setting audio...")
                final_clip = video_clip.set_audio(audio_clip)
                
                logger.info("Writing final video file...")
                temp_output = paths["final"].parent / f"temp_output.mp4"
                
                # ffmpeg-python 직접 사용
                import ffmpeg
                try:
                    final_clip.write_videofile(
                        str(temp_output),
                        codec='libx264',
                        preset='medium',
                        remove_temp=True,
                        write_logfile=True,
                        fps=FPS
                    )
                    
                    # ffmpeg로 오디오 추가
                    stream = ffmpeg.input(str(temp_output))
                    audio_stream = ffmpeg.input(str(paths["audio"]))
                    stream = ffmpeg.output(
                        stream, 
                        audio_stream, 
                        str(paths["final"]),
                        acodec='aac',
                        audio_bitrate='192k',
                        vcodec='copy'  # 비디오는 그대로 복사
                    )
                    ffmpeg.run(stream, overwrite_output=True)
                    
                    temp_output.unlink(missing_ok=True)
                    
                except ffmpeg.Error as e:
                    logger.error(f"FFmpeg error: {e.stderr.decode()}")
                    raise
                
                logger.info("Closing clips...")
                video_clip.close()
                audio_clip.close()

                if not paths["final"].exists():
                    raise Exception("Final video file was not created")
                
                final_size = paths["final"].stat().st_size
                logger.info(f"Final video file created successfully. Size: {final_size / (1024*1024):.2f} MB")
                
                return str(paths["final"])
                
            except Exception as e:
                logger.error(f"Error combining audio and video: {str(e)}")
                if 'video_clip' in locals():
                    video_clip.close()
                if 'audio_clip' in locals():
                    audio_clip.close()
                raise

    except Exception as e:
        logger.error(f"Error during recording save: {str(e)}")
        st.error(f"Error during recording save: {str(e)}")
        return None

def render_record_screen():
    """Main recording screen with UI elements and recording logic."""
    render_navbar()
    create_back_button("information")

    # Initialize global frame sync if not exists
    global frames, recording_state
    if frames is None:
        logger.info("Initializing global frame sync")
        frames = FrameSync()

    # Header
    st.markdown("<h2 style='text-align: center;'>🎥 테스트 영상 녹화</h2>", unsafe_allow_html=True)
    st.write("준비가 되면 아래의 녹화 화면에서 버튼을 눌러 녹화를 시작해주세요.")

    status_container = st.container()

    # Debug information
    with st.expander("Debug Info"):
        if frames:
            st.write("Video Frames:", len(frames.video_frames))
            st.write("Audio Frames:", len(frames.audio_frames))
            st.write("Is Recording:", recording_state["is_recording"])

    try:
        ctx = webrtc_streamer(
            key="recording",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={
                "video": True,
                "audio": True,
            },
            async_processing=True,
        )

        col1, col2 = st.columns(2)
        
        with col1:
            if ctx.state.playing:
                if not recording_state["is_recording"]:
                    if st.button("녹화 시작"):
                        recording_state["is_recording"] = True
                        frames.clear()
                        status_container.info("녹화 중...")
                        logger.info("Started recording")

        with col2:
            if ctx.state.playing and recording_state["is_recording"]:
                if st.button("녹화 완료"):
                    recording_state["is_recording"] = False
                    with st.spinner("녹화 완료 처리 중..."):
                        logger.info("Saving recording...")
                        output_path = save_recording(frames)
                        if output_path:
                            logger.info(f"Recording saved to: {output_path}")
                            st.success("녹화가 완료되었습니다.")
                            st.session_state.recording_complete = True
                            st.session_state.video_path = output_path
                            st.session_state.page = "result"
                            st.experimental_rerun()

        # Status message
        if ctx.state.playing:
            if recording_state["is_recording"]:
                status_container.info("녹화 중...")
            else:
                status_container.info("카메라 준비됨")
        else:
            status_container.warning("카메라가 비활성화되어 있습니다.")

    except Exception as e:
        logger.error("Error in render_record_screen: %s", str(e), exc_info=True)
        st.error(f"Error: {str(e)}")