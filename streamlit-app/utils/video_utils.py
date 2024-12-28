import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64
import json
from typing import Dict, List, Tuple

def get_video_base64(video_path: str) -> str:
    """Encode video file to base64"""
    with open(video_path, 'rb') as f:
        data = f.read()
        return base64.b64encode(data).decode()

def create_attention_indicator_html(max_score: float) -> str:
    """Generate HTML/CSS for attention indicator"""
    return """
        <div style="position: relative;">
            <!-- Attention level gradient bar -->
            <div id="attentionIndicator" style="
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 8px;
                background: linear-gradient(to right, 
                    rgba(220,220,220,0.7) 0%, 
                    rgba(100,149,237,0.7) 50%, 
                    rgba(0,0,139,0.7) 100%
                );
                opacity: 0;
                transition: opacity 0.3s;
                border-radius: 4px;
            "></div>
            
            <!-- Attention score display -->
            <div id="attentionValue" style="
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
            
            <!-- Attention level text display -->
            <div id="attentionText" style="
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

def video_player_with_attention_tracking(
    video_path: str,
    tokens_data: Dict[str, List],
    width: int = 640,
    height: int = 360,
    max_score: float = 1.0
):
    """
    Create video player with attention tracking feature
    
    Parameters:
    video_path: str - Path to video file
    tokens_data: Dict[str, List] - Token data dictionary {'token': [], 'start': [], 'end': [], 'importance': []}
    width: int - Video player width (pixels)
    height: int - Video player height (pixels)
    max_score: float - Maximum attention score
    """
    if video_path and Path(video_path).exists():
        if 'video_time' not in st.session_state:
            st.session_state.video_time = 0.0
        if 'video_score' not in st.session_state:
            st.session_state.video_score = 0.0

        # Generate time-based score data
        score_data = sorted(
            [[float(start), float(end), float(importance)] 
             for start, end, importance in zip(tokens_data['start'], tokens_data['end'], tokens_data['importance'])],
            key=lambda x: x[0]
        )
        score_data_json = json.dumps(score_data)

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
            {create_attention_indicator_html(max_score)}
            
            <script>
                const scoreData = {score_data_json};
                
                function getScore(currentTime) {{
                    let left = 0;
                    let right = scoreData.length - 1;
                    
                    while (left <= right) {{
                        const mid = Math.floor((left + right) / 2);
                        const [start, end, score] = scoreData[mid];
                        
                        if (start <= currentTime && currentTime <= end) {{
                            return score;
                        }} else if (currentTime < start) {{
                            right = mid - 1;
                        }} else {{
                            left = mid + 1;
                        }}
                    }}
                    
                    return 0.0;
                }}

                var video = document.getElementById('myVideo');
                var attentionIndicator = document.getElementById('attentionIndicator');
                var attentionValue = document.getElementById('attentionValue');
                var attentionText = document.getElementById('attentionText');
                let lastTime = 0;
                
                // Initialize elements
                attentionValue.textContent = 'Attention: 0.00';
                attentionText.textContent = 'Normal';
                attentionText.style.backgroundColor = 'rgba(220,220,220,0.8)';
                
                function updateAttentionVisualization(time) {{
                    const score = getScore(time);
                    const normalizedScore = score / {max_score};
                    
                    attentionIndicator.style.opacity = normalizedScore;
                    attentionValue.textContent = `Attention: ${{score.toFixed(2)}}`;
                    
                    if (score < {max_score/3}) {{
                        attentionValue.style.backgroundColor = 'rgba(220,220,220,0.8)';
                        attentionText.textContent = 'Normal';
                        attentionText.style.backgroundColor = 'rgba(220,220,220,0.8)';
                    }} else if (score < {max_score*2/3}) {{
                        attentionValue.style.backgroundColor = 'rgba(100,149,237,0.8)';
                        attentionText.textContent = 'Notable';
                        attentionText.style.backgroundColor = 'rgba(100,149,237,0.8)';
                    }} else {{
                        attentionValue.style.backgroundColor = 'rgba(0,0,139,0.8)';
                        attentionText.textContent = 'Focus';
                        attentionText.style.backgroundColor = 'rgba(0,0,139,0.8)';
                    }}
                }}
                
                function sendTimeToStreamlit(currentTime) {{
                    if (currentTime !== lastTime) {{
                        lastTime = currentTime;
                        updateAttentionVisualization(currentTime);
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

        value = components.html(html_code, height=height + 100)
        
        try:
            if isinstance(value, dict) and 'time' in value:
                current_time = float(value['time'])
                current_score = getScore(current_time, score_data)
                
                st.session_state.video_time = current_time
                st.session_state.video_score = current_score
                
                return {
                    "time": current_time,
                    "score": current_score
                }
        except Exception as e:
            st.error(f"Error processing time update: {str(e)}")
            
        return {
            "time": st.session_state.video_time,
            "score": st.session_state.video_score
        }

def getScore(current_time: float, score_data: List[List[float]]) -> float:
    left, right = 0, len(score_data) - 1
    while left <= right:
        mid = (left + right) // 2
        start, end, score = score_data[mid]
        
        if start <= current_time <= end:
            return score
        elif current_time < start:
            right = mid - 1
        else:
            left = mid + 1
            
    return 0.0

def render_attention_box(attention_placeholder, attention_level: float, max_attention: float = 1.0):
    if attention_level > max_attention * 0.66:
        attention_placeholder.info("🔍 High Focus Area: Model shows high attention to speech in this segment.")
    elif attention_level > max_attention * 0.33:
        attention_placeholder.info("👀 Notable Area: Model is closely analyzing speech in this segment.")
    elif attention_level > 0:
        attention_placeholder.info("ℹ️ Normal Area: Model is performing baseline analysis.")