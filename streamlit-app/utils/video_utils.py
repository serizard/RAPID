import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class ImportanceScoreManager:
    def __init__(self):
        self._cache: Dict[float, float] = {}
        self._sorted_timestamps: List[Tuple[float, float]] = []
    
    def initialize(self, tokens_df: pd.DataFrame):
        """Initialize the manager with tokens dataframe"""
        # Convert to sorted list of tuples for efficient lookup
        self._sorted_timestamps = sorted(
            zip(tokens_df['start'], tokens_df['end'], tokens_df['importance']),
            key=lambda x: x[0]
        )
        self._cache.clear()
    
    def get_score(self, current_time: float) -> float:
        """Get importance score for current timestamp with caching"""
        # Check cache first
        if current_time in self._cache:
            return self._cache[current_time]
        
        # Binary search for the relevant time segment
        left, right = 0, len(self._sorted_timestamps) - 1
        while left <= right:
            mid = (left + right) // 2
            start, end, score = self._sorted_timestamps[mid]
            
            if start <= current_time <= end:
                self._cache[current_time] = score
                return score
            elif current_time < start:
                right = mid - 1
            else:
                left = mid + 1
                
        return 0.0  # Default score if no matching segment found

def get_video_base64(video_path: str) -> str:
    with open(video_path, 'rb') as f:
        data = f.read()
        return base64.b64encode(data).decode()

def create_risk_indicator_html(max_score: float) -> str:
    """
    Create HTML/CSS for the risk indicator that shows aphasia risk level
    Returns HTML string with two components:
    1. A gradient bar at the top that changes opacity based on risk
    2. A text indicator showing the numeric risk score
    """
    return """
        <div style="position: relative;">
            <!-- Risk level gradient bar -->
            <div id="riskIndicator" style="
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 8px;
                background: linear-gradient(to right, 
                    rgba(0,255,0,0.7) 0%, 
                    rgba(255,255,0,0.7) 50%, 
                    rgba(255,0,0,0.7) 100%
                );
                opacity: 0;
                transition: opacity 0.3s;
                border-radius: 4px;
            "></div>
            
            <!-- Risk score display -->
            <div id="riskValue" style="
                position: absolute;
                top: 15px;
                right: 15px;
                padding: 8px 12px;
                background: rgba(0,0,0,0.7);
                color: white;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: background-color 0.3s;
                z-index: 1000;
            "></div>
            
            <!-- Risk level text display -->
            <div id="riskText" style="
                position: absolute;
                top: 15px;
                left: 15px;
                padding: 8px 12px;
                color: white;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                opacity: 0.9;
                transition: opacity 0.3s;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                z-index: 1000;
            "></div>
        </div>
    """

def video_player_with_risk_tracking(
    video_path: str,
    score_manager: ImportanceScoreManager,
    max_score: float = 1.0,
    width: int = 640,      # 픽셀 단위의 너비
    height: int = 360      # 픽셀 단위의 높이
):
    if video_path and Path(video_path).exists():
        if 'video_time' not in st.session_state:
            st.session_state.video_time = 0.0
        if 'video_score' not in st.session_state:
            st.session_state.video_score = 0.0

        video_base64 = get_video_base64(video_path)
        
        html_code = f"""
        <div style="position: relative; width: {width}px; margin: 0 auto;">
            <video 
                id="myVideo" 
                style="width: {width}px; height: {height}px; object-fit: contain;"
                controls
            >
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            {create_risk_indicator_html(max_score)}
            
            <script>
                var video = document.getElementById('myVideo');
                var riskIndicator = document.getElementById('riskIndicator');
                var riskValue = document.getElementById('riskValue');
                var riskText = document.getElementById('riskText');
                let lastTime = 0;
                
                // Initialize elements
                riskValue.textContent = '위험도: 0.00';
                riskText.textContent = '정상';
                riskText.style.backgroundColor = 'rgba(0,128,0,0.8)';
                
                function updateRiskVisualization(time) {{
                    const score = {max_score} * (1 - Math.abs(Math.sin(time / 2))); // 임시 점수 계산
                    const normalizedScore = score / {max_score};
                    
                    // Update gradient bar opacity
                    riskIndicator.style.opacity = normalizedScore;
                    
                    // Update numeric score
                    riskValue.textContent = `위험도: ${{score.toFixed(2)}}`;
                    
                    // Update colors and text based on risk level
                    if (score < {max_score/3}) {{
                        riskValue.style.backgroundColor = 'rgba(0,128,0,0.8)';
                        riskText.textContent = '정상';
                        riskText.style.backgroundColor = 'rgba(0,128,0,0.8)';
                    }} else if (score < {max_score*2/3}) {{
                        riskValue.style.backgroundColor = 'rgba(255,165,0,0.8)';
                        riskText.textContent = '주의';
                        riskText.style.backgroundColor = 'rgba(255,165,0,0.8)';
                    }} else {{
                        riskValue.style.backgroundColor = 'rgba(255,0,0,0.8)';
                        riskText.textContent = '위험';
                        riskText.style.backgroundColor = 'rgba(255,0,0,0.8)';
                    }}
                }}
                
                function sendTimeToStreamlit(currentTime) {{
                    if (currentTime !== lastTime) {{
                        lastTime = currentTime;
                        updateRiskVisualization(currentTime);
                        window.Streamlit.setComponentValue({{
                            time: currentTime,
                            type: 'time_update'
                        }});
                    }}
                }}
                
                function updateTime() {{
                    sendTimeToStreamlit(video.currentTime);
                }}
                
                video.ontimeupdate = updateTime;
                video.addEventListener('play', updateTime);
                video.addEventListener('pause', updateTime);
                video.addEventListener('seeked', updateTime);
                
                setInterval(updateTime, 100);
                
                window.Streamlit.setComponentReady();
            </script>
        </div>
        """

        # Render video component with more extra space
        value = components.html(html_code, height=height + 100)  # 컨트롤러와 위험도 표시를 위한 여유 공간 증가
        
        try:
            if isinstance(value, dict) and 'time' in value:
                current_time = float(value['time'])
                current_score = score_manager.get_score(current_time)
                
                st.session_state.video_time = current_time
                st.session_state.video_score = current_score
                
                return {
                    "time": current_time,
                    "score": current_score
                }
        except Exception as e:
            st.error(f"Error processing time update: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            
        # Return last known values from session state
        return {
            "time": st.session_state.video_time,
            "score": st.session_state.video_score
        }


def render_warning_box(warning_placeholder, risk_level: float, max_risk: float = 1.0):
    """Display a warning box based on the current risk level"""
    if risk_level > max_risk * 0.66:
        warning_placeholder.error("⚠️ 높은 위험 구간: 해당 구간에서 실어증 징후가 강하게 감지되었습니다.")
    elif risk_level > max_risk * 0.33:
        warning_placeholder.warning("⚠️ 중간 위험 구간: 실어증 관련 특징이 일부 감지되었습니다.")
    elif risk_level > 0:
        warning_placeholder.info("ℹ️ 낮은 위험 구간: 정상 범위 내 발화가 감지되었습니다.")