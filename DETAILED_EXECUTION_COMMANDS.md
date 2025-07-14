# S-Class DMS v19.0 - 상세 실행 명령어 가이드

## 🛠️ Phase 5: MediaPipe Manager 통합 - 실행 명령어

### **Step 5.1: 백업 생성 및 레거시 파일 삭제**

```bash
# Git 백업 브랜치 생성
git checkout -b phase5-mediapipe-consolidation
git add -A
git commit -m "Phase 5 시작: MediaPipe Manager 통합 전 백업"

# 레거시 파일 삭제
git rm systems/mediapipe_manager.py
```

### **Step 5.2: 개선된 파일 이름 변경**

```bash
# v2 파일을 표준 이름으로 변경
git mv systems/mediapipe_manager_v2.py systems/mediapipe_manager.py
```

### **Step 5.3: Import 참조 업데이트 (Copy-Paste 명령어)**

```bash
# 현재 import 상태 확인
echo "=== 현재 MediaPipe Manager Import 상태 확인 ==="
grep -rn "mediapipe_manager" --include="*.py" . | grep -v ".git"

# v2 참조가 있는지 확인
echo "=== v2 참조 확인 ==="
grep -rn "mediapipe_manager_v2" --include="*.py" .

# app.py에서 import 업데이트 (이미 정확함 - 확인용)
echo "=== app.py import 확인 ==="
grep -n "from systems.mediapipe_manager import" app.py

# app_.py에서 import 업데이트 (이미 정확함 - 확인용)
echo "=== app_.py import 확인 ==="
grep -n "from systems.mediapipe_manager import" app_.py
```

### **Step 5.4: 기능 보존 검증 명령어**

```bash
# 통합된 파일의 주요 클래스 확인
echo "=== MediaPipe Manager 주요 클래스 확인 ==="
grep -n "class.*MediaPipe" systems/mediapipe_manager.py

# 동적 로딩/언로딩 기능 확인
echo "=== 동적 모델 로딩 기능 확인 ==="
grep -n "def.*load\|def.*unload\|def.*dynamic" systems/mediapipe_manager.py

# MediaPipe Tasks API 버전 확인
echo "=== Tasks API 버전 확인 ==="
grep -n "Tasks.*API\|0\.10\|mediapipe.*task" systems/mediapipe_manager.py

# 에러 처리 및 복구 로직 확인
echo "=== 에러 처리 로직 확인 ==="
grep -n "try:\|except:\|recovery\|failover" systems/mediapipe_manager.py

# 통합 완료 커밋
git add systems/mediapipe_manager.py
git commit -m "Phase 5 완료: MediaPipe Manager v2 기능을 표준 파일로 통합"
```

---

## 🔧 Phase 6: 메인 애플리케이션 통합 - 실행 명령어

### **Step 6.1: 혁신 기능 마이그레이션 명령어**

#### **6.1.1: 5대 혁신 기능 코드 식별**

```bash
# 백업 브랜치 생성
git checkout -b phase6-main-application-integration
git add -A
git commit -m "Phase 6 시작: 메인 애플리케이션 통합 전 백업"

echo "=== 5대 혁신 기능 Import 문 확인 ==="
grep -n "from systems\." s_class_dms_v19_main.py

echo "=== AI Driving Coach 관련 코드 식별 ==="
grep -n "AIDrivingCoach\|ai_coach\|driving.*coach" s_class_dms_v19_main.py

echo "=== V2D Healthcare 관련 코드 식별 ==="
grep -n "V2DHealthcareSystem\|healthcare\|biometric\|health.*monitoring" s_class_dms_v19_main.py

echo "=== AR HUD System 관련 코드 식별 ==="
grep -n "ARHUDSystem\|ar_hud\|augmented.*reality\|VehicleContext" s_class_dms_v19_main.py

echo "=== Emotional Care System 관련 코드 식별 ==="
grep -n "EmotionalCareSystem\|emotional_care\|emotion.*recognition" s_class_dms_v19_main.py

echo "=== Digital Twin Platform 관련 코드 식별 ==="
grep -n "DigitalTwinPlatform\|digital_twin\|virtual.*driving\|simulation" s_class_dms_v19_main.py
```

#### **6.1.2: SClassDMSv19 클래스 구조 분석**

```bash
echo "=== SClassDMSv19 클래스 메서드 구조 ==="
grep -n "def.*" s_class_dms_v19_main.py | head -20

echo "=== 혁신 시스템 초기화 코드 확인 ==="
sed -n '102,147p' s_class_dms_v19_main.py

echo "=== 메인 루프 처리 로직 확인 ==="
sed -n '222,268p' s_class_dms_v19_main.py

echo "=== 혁신 시스템 데이터 처리 로직 확인 ==="
sed -n '269,326p' s_class_dms_v19_main.py
```

### **Step 6.2: 레거시 로직 분석 명령어**

#### **6.2.1: app.py와 app_.py의 중요 기능 식별**

```bash
echo "=== app.py GUI 관련 로직 확인 ==="
grep -n "GUI\|tkinter\|Qt\|interface\|window\|dialog" app.py

echo "=== app.py CLI 인수 처리 확인 ==="
grep -n "argparse\|argv\|argument\|parser" app.py

echo "=== app.py 초기화 로직 확인 ==="
grep -n "def.*init\|def.*setup\|def.*configure\|DMSApp" app.py | head -10

echo "=== app.py MediaPipe Manager 사용 확인 ==="
grep -n "mediapipe_manager\|EnhancedMediaPipeManager" app.py

echo "=== app_.py 고유 기능 확인 ==="
grep -n "def.*" app_.py | grep -v "def.*init\|def.*setup" | head -10

echo "=== app_.py 진단 로깅 기능 확인 ==="
grep -n "진단\|logger\.info.*진단" app_.py | head -10
```

#### **6.2.2: main.py의 GUI 시스템 분석**

```bash
echo "=== main.py GUI 클래스 구조 확인 ==="
grep -n "class.*" main.py

echo "=== main.py S-Class 기능 설정 확인 ==="
grep -n "enable_.*\|system_type\|sclass" main.py

echo "=== main.py 터미널 모드 기능 확인 ==="
grep -n "def.*terminal\|terminal.*mode\|no.*gui" main.py

echo "=== main.py 메인 함수 확인 ==="
sed -n '885,954p' main.py
```

### **Step 6.3: 통합 계획 실행**

#### **6.3.1: s_class_dms_v19_main.py에서 main.py로 핵심 기능 이전**

**필요한 Import 문들을 main.py에 추가:**

```python
# main.py 상단에 추가할 Import 문들
# (이 코드는 실제 edit_file로 적용해야 함)

# 5대 혁신 기능 Import 추가
from systems.ai_driving_coach import AIDrivingCoach
from systems.v2d_healthcare import V2DHealthcareSystem  
from systems.ar_hud_system import ARHUDSystem, VehicleContext
from systems.emotional_care_system import EmotionalCareSystem
from systems.digital_twin_platform import DigitalTwinPlatform

# S-Class v19 관련 Import 추가
from config.settings import get_config, FeatureFlagConfig
from models.data_structures import UIState
from io_handler.ui import UIHandler
```

**SClassDMSv19 클래스를 main.py에 통합:**

```python
# main.py에 추가할 SClassDMSv19 클래스 (의사코드)
class SClassDMSv19Enhanced(SClass_DMS_GUI_Setup):
    """S-Class DMS v19.0 GUI와 혁신 기능 통합 클래스"""
    
    def __init__(self, root):
        super().__init__(root)
        
        # v19 혁신 기능 초기화
        self.innovation_systems = {}
        self.feature_flags = FeatureFlagConfig(edition="RESEARCH")
        
        # 5대 혁신 시스템 초기화
        self._initialize_innovation_systems()
    
    def _initialize_innovation_systems(self):
        """5대 혁신 시스템 초기화"""
        # AI Driving Coach
        self.innovation_systems["ai_coach"] = AIDrivingCoach("default")
        
        # V2D Healthcare
        self.innovation_systems["healthcare"] = V2DHealthcareSystem("default")
        
        # AR HUD System  
        self.innovation_systems["ar_hud"] = ARHUDSystem()
        
        # Emotional Care System
        self.innovation_systems["emotional_care"] = EmotionalCareSystem("default")
        
        # Digital Twin Platform
        self.innovation_systems["digital_twin"] = DigitalTwinPlatform()
```

#### **6.3.2: 통합 실행 명령어**

```bash
# 백업 생성
cp main.py main_backup_$(date +%Y%m%d_%H%M%S).py
cp app.py app_backup_$(date +%Y%m%d_%H%M%S).py
cp app_.py app__backup_$(date +%Y%m%d_%H%M%S).py

echo "=== 통합 전 파일 크기 확인 ==="
wc -l main.py app.py app_.py s_class_dms_v19_main.py

# 통합 완료 후 레거시 파일 삭제
echo "=== 통합 완료 후 정리 명령어 (주의: 통합 완료 후에만 실행) ==="
echo "git rm app.py app_.py s_class_dms_v19_main.py"
echo "git add main.py"
echo "git commit -m 'Phase 6 완료: 5대 혁신 기능을 main.py로 통합, 레거시 파일 정리'"
```

### **Step 6.4: 검증 명령어**

#### **6.4.1: 통합된 main.py 검증**

```bash
echo "=== 통합된 main.py 구조 검증 ==="
grep -n "class.*" main.py

echo "=== 5대 혁신 기능 Import 확인 ==="
grep -n "from systems\.*" main.py

echo "=== 혁신 시스템 초기화 코드 확인 ==="
grep -n "innovation.*systems\|AIDrivingCoach\|V2DHealthcareSystem\|ARHUDSystem\|EmotionalCareSystem\|DigitalTwinPlatform" main.py

echo "=== GUI 호환성 유지 확인 ==="
grep -n "SClass_DMS_GUI_Setup\|tkinter\|ttk" main.py

echo "=== 통합 후 파일 크기 확인 ==="
wc -l main.py

echo "=== 문법 오류 확인 ==="
python -m py_compile main.py && echo "✅ 문법 오류 없음" || echo "❌ 문법 오류 발견"
```

#### **6.4.2: 기능 테스트 명령어**

```bash
echo "=== 기본 실행 테스트 ==="
python main.py --help

echo "=== GUI 모드 테스트 (백그라운드) ==="
timeout 10 python main.py &
sleep 5
pkill -f "python main.py"

echo "=== 혁신 기능 개별 테스트 (예시) ==="
python -c "
try:
    from systems.ai_driving_coach import AIDrivingCoach
    from systems.v2d_healthcare import V2DHealthcareSystem
    from systems.ar_hud_system import ARHUDSystem
    from systems.emotional_care_system import EmotionalCareSystem
    from systems.digital_twin_platform import DigitalTwinPlatform
    print('✅ 모든 혁신 시스템 Import 성공')
except Exception as e:
    print(f'❌ Import 오류: {e}')
"
```

---

## 📊 통합 완료 검증 체크리스트

### **Phase 5 검증 체크리스트**

```bash
# Phase 5 완료 확인 스크립트
echo "=== Phase 5 완료 검증 ==="

# 1. mediapipe_manager.py만 존재하는지 확인
if [ -f "systems/mediapipe_manager.py" ] && [ ! -f "systems/mediapipe_manager_v2.py" ]; then
    echo "✅ MediaPipe Manager 파일 통합 완료"
else
    echo "❌ MediaPipe Manager 파일 통합 미완료"
fi

# 2. 파일 크기 확인 (v2 기능 포함되어야 함)
file_size=$(wc -l < systems/mediapipe_manager.py)
if [ $file_size -gt 400 ]; then
    echo "✅ MediaPipe Manager 크기 적절 ($file_size lines)"
else
    echo "❌ MediaPipe Manager 크기 부족 ($file_size lines)"
fi

# 3. Import 참조 확인
if grep -q "from systems.mediapipe_manager import" app.py app_.py; then
    echo "✅ Import 참조 정상"
else
    echo "❌ Import 참조 문제"
fi

# 4. MediaPipe Tasks API 확인
if grep -q "Tasks.*API\|0\.10" systems/mediapipe_manager.py; then
    echo "✅ 최신 MediaPipe Tasks API 포함"
else
    echo "❌ 최신 MediaPipe Tasks API 누락"
fi
```

### **Phase 6 검증 체크리스트**

```bash
# Phase 6 완료 확인 스크립트
echo "=== Phase 6 완료 검증 ==="

# 1. 레거시 파일들이 삭제되었는지 확인
legacy_files_exist=false
for file in app.py app_.py s_class_dms_v19_main.py; do
    if [ -f "$file" ]; then
        echo "❌ 레거시 파일 $file 여전히 존재"
        legacy_files_exist=true
    fi
done

if [ "$legacy_files_exist" = false ]; then
    echo "✅ 모든 레거시 파일 정리 완료"
fi

# 2. main.py에 5대 혁신 기능 Import 확인
innovation_imports=0
for system in "AIDrivingCoach" "V2DHealthcareSystem" "ARHUDSystem" "EmotionalCareSystem" "DigitalTwinPlatform"; do
    if grep -q "$system" main.py; then
        ((innovation_imports++))
    fi
done

if [ $innovation_imports -eq 5 ]; then
    echo "✅ 5대 혁신 기능 모두 Import됨"
else
    echo "❌ 혁신 기능 Import 불완전 ($innovation_imports/5)"
fi

# 3. GUI 기능 유지 확인
if grep -q "SClass_DMS_GUI_Setup\|tkinter" main.py; then
    echo "✅ GUI 기능 유지됨"
else
    echo "❌ GUI 기능 누락"
fi

# 4. 파일 크기 확인 (통합으로 증가해야 함)
main_size=$(wc -l < main.py)
if [ $main_size -gt 1200 ]; then
    echo "✅ main.py 크기 적절 ($main_size lines)"
else
    echo "❌ main.py 크기 부족 ($main_size lines)"
fi

# 5. 문법 오류 확인
if python -m py_compile main.py 2>/dev/null; then
    echo "✅ main.py 문법 오류 없음"
else
    echo "❌ main.py 문법 오류 발견"
fi
```

---

## 🚀 최종 실행 순서

### **전체 통합 실행 스크립트**

```bash
#!/bin/bash
# S-Class DMS v19.0 Phase 5-6 통합 실행 스크립트

set -e  # 오류 시 스크립트 중단

echo "🚀 S-Class DMS v19.0 Phase 5-6 통합 시작"
echo "=========================================="

# Phase 5: MediaPipe Manager 통합
echo "📊 Phase 5: MediaPipe Manager 통합 시작..."

# 백업 생성
git checkout -b phase5-6-consolidation
git add -A
git commit -m "Phase 5-6 통합 시작: 전체 백업"

# MediaPipe Manager 통합
git rm systems/mediapipe_manager.py
git mv systems/mediapipe_manager_v2.py systems/mediapipe_manager.py
git add systems/mediapipe_manager.py
git commit -m "Phase 5 완료: MediaPipe Manager 통합"

echo "✅ Phase 5 완료!"

# Phase 6: 메인 애플리케이션 통합
echo "🔧 Phase 6: 메인 애플리케이션 통합 시작..."

# 백업 생성
cp main.py main_backup_$(date +%Y%m%d_%H%M%S).py
cp app.py app_backup_$(date +%Y%m%d_%H%M%S).py
cp app_.py app__backup_$(date +%Y%m%d_%H%M%S).py

echo "⚠️  수동 작업 필요:"
echo "1. s_class_dms_v19_main.py의 5대 혁신 기능을 main.py에 통합"
echo "2. app.py와 app_.py의 중요한 레거시 로직을 main.py에 병합"
echo "3. 통합 완료 후 다음 명령어 실행:"
echo "   git rm app.py app_.py s_class_dms_v19_main.py"
echo "   git add main.py"
echo "   git commit -m 'Phase 6 완료: 5대 혁신 기능 통합'"

echo "📋 다음 단계: DETAILED_EXECUTION_COMMANDS.md의 Step 6.3 참조"
```

---

*상세 실행 명령어 가이드 생성일: 2025-01-15*  
*S-Class DMS v19.0 "The Final Integration Commands"*  
*Copy-Paste 준비된 실행 가능한 명령어 모음*