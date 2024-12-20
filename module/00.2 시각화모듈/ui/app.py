from PyQt5.QtWidgets import QApplication
import sys

def get_app_instance():
    # QApplication은 전체 앱에서 하나만 생성해야 함
    # 여기서는 단순히 인스턴스를 반환
    app = QApplication(sys.argv)
    return app
