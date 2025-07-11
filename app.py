#!/usr/bin/env python3
"""
🚀 S-Class DMS v19.0 - API 서버
웹 기반 대시보드 및 API 엔드포인트 제공
"""

from flask import Flask, jsonify, request, render_template_string
import asyncio
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# S-Class DMS v19 모듈
from s_class_dms_v19_main import SClassDMSv19

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 앱 생성
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sclass-dms-v19-secret'

# 전역 DMS 시스템
dms_system: Optional[SClassDMSv19] = None
system_thread: Optional[threading.Thread] = None
is_running = False


# HTML 템플릿 (간단한 웹 대시보드)
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🚀 S-Class DMS v19.0 - Web Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 15px;
            border: 1px solid #00d4ff;
        }
        .title {
            color: #00d4ff;
            font-size: 2.5em;
            margin: 0;
            text-shadow: 0 0 20px #00d4ff;
        }
        .subtitle {
            color: #8a8a8a;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #00d4ff;
            transition: transform 0.3s ease;
        }
        .status-card:hover {
            transform: translateY(-5px);
        }
        .card-title {
            color: #00d4ff;
            font-size: 1.3em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            color: #ffffff;
        }
        .metric-value {
            color: #00ff9f;
            font-weight: bold;
        }
        .controls {
            text-align: center;
            margin: 30px 0;
        }
        .btn {
            background: linear-gradient(45deg, #00d4ff, #00ff9f);
            color: #1a1a2e;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 1.1em;
            font-weight: bold;
            margin: 0 10px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
        }
        .btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        .systems-status {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .system-indicator {
            background: rgba(22, 33, 62, 0.6);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #00d4ff;
        }
        .system-name {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .status-active {
            color: #00ff9f;
        }
        .status-inactive {
            color: #ff6b35;
        }
        .auto-refresh {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 212, 255, 0.2);
            padding: 10px 15px;
            border-radius: 20px;
            border: 1px solid #00d4ff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="auto-refresh">
            <small>🔄 Auto-refresh: 5s</small>
        </div>
        
        <div class="header">
            <h1 class="title">🚀 S-Class DMS v19.0</h1>
            <p class="subtitle">차세대 지능형 운전자 모니터링 시스템 • 실시간 웹 대시보드</p>
        </div>

        <div class="controls">
            <button class="btn" onclick="startSystem()" id="startBtn">🚀 시스템 시작</button>
            <button class="btn" onclick="stopSystem()" id="stopBtn" disabled>⏹ 시스템 중지</button>
            <button class="btn" onclick="refreshStatus()">🔄 상태 새로고침</button>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <div class="card-title">📊 시스템 상태</div>
                <div class="metric">
                    <span class="metric-label">실행 상태</span>
                    <span class="metric-value" id="runningStatus">준비</span>
                </div>
                <div class="metric">
                    <span class="metric-label">사용자 ID</span>
                    <span class="metric-value" id="userId">default</span>
                </div>
                <div class="metric">
                    <span class="metric-label">에디션</span>
                    <span class="metric-value" id="edition">RESEARCH</span>
                </div>
                <div class="metric">
                    <span class="metric-label">업타임</span>
                    <span class="metric-value" id="uptime">00:00:00</span>
                </div>
            </div>

            <div class="status-card">
                <div class="card-title">⚡ 성능 메트릭</div>
                <div class="metric">
                    <span class="metric-label">FPS</span>
                    <span class="metric-value" id="fps">0.0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">처리 시간</span>
                    <span class="metric-value" id="processTime">0ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">메모리 사용량</span>
                    <span class="metric-value" id="memory">0MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">세션 프레임</span>
                    <span class="metric-value" id="sessionFrames">0</span>
                </div>
            </div>

            <div class="status-card">
                <div class="card-title">🧠 5대 혁신 시스템</div>
                <div class="systems-status">
                    <div class="system-indicator">
                        <div class="system-name">🎓 AI 코치</div>
                        <div class="status-inactive" id="aiCoachStatus">비활성</div>
                    </div>
                    <div class="system-indicator">
                        <div class="system-name">🏥 헬스케어</div>
                        <div class="status-inactive" id="healthcareStatus">비활성</div>
                    </div>
                    <div class="system-indicator">
                        <div class="system-name">🥽 AR HUD</div>
                        <div class="status-inactive" id="arHudStatus">비활성</div>
                    </div>
                    <div class="system-indicator">
                        <div class="system-name">🎭 감성 케어</div>
                        <div class="status-inactive" id="emotionalCareStatus">비활성</div>
                    </div>
                    <div class="system-indicator">
                        <div class="system-name">🤖 디지털 트윈</div>
                        <div class="status-inactive" id="digitalTwinStatus">비활성</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let refreshInterval;
        
        function startSystem() {
            fetch('/api/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        startAutoRefresh();
                    }
                    alert(data.message);
                });
        }
        
        function stopSystem() {
            fetch('/api/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        stopAutoRefresh();
                    }
                    alert(data.message);
                });
        }
        
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateUI(data));
        }
        
        function updateUI(status) {
            document.getElementById('runningStatus').textContent = status.is_running ? '실행 중' : '중지됨';
            document.getElementById('userId').textContent = status.user_id || 'default';
            document.getElementById('edition').textContent = status.edition || 'RESEARCH';
            
            // 성능 메트릭
            if (status.performance) {
                document.getElementById('fps').textContent = (status.performance.fps || 0).toFixed(1);
                document.getElementById('processTime').textContent = `${status.performance.process_time || 0}ms`;
                document.getElementById('memory').textContent = `${status.performance.memory || 0}MB`;
                document.getElementById('sessionFrames').textContent = status.performance.session_frames || 0;
            }
            
            // 시스템 상태
            if (status.systems) {
                updateSystemStatus('aiCoachStatus', status.systems.ai_coach_active);
                updateSystemStatus('healthcareStatus', status.systems.healthcare_active);
                updateSystemStatus('arHudStatus', status.systems.ar_hud_active);
                updateSystemStatus('emotionalCareStatus', status.systems.emotional_care_active);
                updateSystemStatus('digitalTwinStatus', status.systems.digital_twin_active);
            }
        }
        
        function updateSystemStatus(elementId, isActive) {
            const element = document.getElementById(elementId);
            if (isActive) {
                element.textContent = '활성';
                element.className = 'status-active';
            } else {
                element.textContent = '비활성';
                element.className = 'status-inactive';
            }
        }
        
        function startAutoRefresh() {
            refreshInterval = setInterval(refreshStatus, 5000);
        }
        
        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        }
        
        // 초기 상태 로드
        refreshStatus();
        
        // 페이지 로드 시 자동 새로고침 시작
        startAutoRefresh();
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """웹 대시보드 메인 페이지"""
    return render_template_string(DASHBOARD_TEMPLATE)


@app.route('/api/status')
def get_status():
    """시스템 상태 API"""
    global dms_system, is_running
    
    status = {
        'is_running': is_running,
        'user_id': dms_system.user_id if dms_system else 'default',
        'edition': dms_system.edition if dms_system else 'RESEARCH',
        'timestamp': datetime.now().isoformat()
    }
    
    if dms_system and is_running:
        # 시스템 상태
        system_status = dms_system.get_system_status()
        status.update({
            'systems': {
                'ai_coach_active': system_status.get('ai_coach_active', False),
                'healthcare_active': system_status.get('healthcare_active', False),
                'ar_hud_active': system_status.get('ar_hud_active', False),
                'emotional_care_active': system_status.get('emotional_care_active', False),
                'digital_twin_active': system_status.get('digital_twin_active', False)
            },
            'performance': {
                'fps': system_status.get('fps', 0.0),
                'process_time': system_status.get('avg_process_time_ms', 0),
                'memory': system_status.get('memory_usage_mb', 0),
                'session_frames': system_status.get('session_frames', 0)
            }
        })
    else:
        status.update({
            'systems': {
                'ai_coach_active': False,
                'healthcare_active': False,
                'ar_hud_active': False,
                'emotional_care_active': False,
                'digital_twin_active': False
            },
            'performance': {
                'fps': 0.0,
                'process_time': 0,
                'memory': 0,
                'session_frames': 0
            }
        })
    
    return jsonify(status)


@app.route('/api/start', methods=['POST'])
def start_system():
    """시스템 시작 API"""
    global dms_system, system_thread, is_running
    
    if is_running:
        return jsonify({
            'success': False,
            'message': '시스템이 이미 실행 중입니다.'
        })
    
    try:
        # 요청 데이터 파싱
        data = request.get_json() or {}
        user_id = data.get('user_id', 'web_user')
        edition = data.get('edition', 'RESEARCH')
        
        # DMS 시스템 초기화
        dms_system = SClassDMSv19(user_id=user_id, edition=edition)
        
        # 별도 스레드에서 실행
        system_thread = threading.Thread(
            target=run_system_async,
            daemon=True
        )
        system_thread.start()
        
        is_running = True
        
        logger.info(f"S-Class DMS v19 시스템 시작됨 - 사용자: {user_id}, 에디션: {edition}")
        
        return jsonify({
            'success': True,
            'message': 'S-Class DMS v19 시스템이 성공적으로 시작되었습니다.'
        })
        
    except Exception as e:
        logger.error(f"시스템 시작 실패: {e}")
        return jsonify({
            'success': False,
            'message': f'시스템 시작에 실패했습니다: {str(e)}'
        })


@app.route('/api/stop', methods=['POST'])
def stop_system():
    """시스템 중지 API"""
    global dms_system, is_running
    
    if not is_running:
        return jsonify({
            'success': False,
            'message': '시스템이 실행 중이 아닙니다.'
        })
    
    try:
        is_running = False
        
        if dms_system:
            # 비동기 중지 실행
            asyncio.run_coroutine_threadsafe(
                dms_system.stop_system(),
                asyncio.new_event_loop()
            )
        
        logger.info("S-Class DMS v19 시스템 중지됨")
        
        return jsonify({
            'success': True,
            'message': 'S-Class DMS v19 시스템이 정상적으로 중지되었습니다.'
        })
        
    except Exception as e:
        logger.error(f"시스템 중지 실패: {e}")
        return jsonify({
            'success': False,
            'message': f'시스템 중지에 실패했습니다: {str(e)}'
        })


@app.route('/api/config', methods=['GET', 'POST'])
def config_api():
    """설정 관리 API"""
    global dms_system
    
    if request.method == 'GET':
        # 현재 설정 반환
        if dms_system:
            return jsonify({
                'user_id': dms_system.user_id,
                'edition': dms_system.edition,
                'feature_flags': {
                    'ai_coach': True,
                    'healthcare': True,
                    'ar_hud': True,
                    'emotional_care': True,
                    'digital_twin': True
                }
            })
        else:
            return jsonify({
                'user_id': 'default',
                'edition': 'RESEARCH',
                'feature_flags': {}
            })
    
    elif request.method == 'POST':
        # 설정 업데이트
        data = request.get_json()
        
        return jsonify({
            'success': True,
            'message': '설정이 업데이트되었습니다.',
            'config': data
        })


def run_system_async():
    """비동기 시스템 실행"""
    global dms_system, is_running
    
    try:
        # 새 이벤트 루프 생성
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 시스템 실행
        loop.run_until_complete(async_main())
        
    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {e}")
    finally:
        is_running = False


async def async_main():
    """메인 비동기 실행 함수"""
    global dms_system, is_running
    
    try:
        # 시스템 시작
        if await dms_system.start_system():
            logger.info("모든 혁신 시스템이 성공적으로 시작되었습니다")
            
            # 메인 루프 실행
            await dms_system.run_main_loop()
        else:
            logger.error("시스템 시작에 실패했습니다")
            
    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {e}")


def main():
    """메인 실행 함수"""
    print("""
🚀 S-Class DMS v19.0 API 서버 시작
====================================

📱 웹 대시보드: http://localhost:5000
📊 API 엔드포인트:
  GET  /api/status     - 시스템 상태 조회
  POST /api/start      - 시스템 시작
  POST /api/stop       - 시스템 중지
  GET  /api/config     - 설정 조회
  POST /api/config     - 설정 업데이트

💡 사용법:
  웹 브라우저에서 http://localhost:5000 을 열어
  실시간 대시보드로 S-Class DMS v19를 제어하세요.

⏹ 종료: Ctrl+C
    """)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 S-Class DMS v19 API 서버가 종료되었습니다.")


if __name__ == '__main__':
    main()
