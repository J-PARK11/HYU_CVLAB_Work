import os

def rename(parent_folder):

    # 기준 폴더 안에 있는 모든 폴더 목록 가져오기
    folder_list = os.listdir(parent_folder)

    # 모든 폴더에 접미사를 추가하여 이름을 변경
    for folder in folder_list:
        new_folder_name = folder[-6:] + '_' + folder[:-6]
        os.rename(os.path.join(parent_folder, folder), os.path.join(parent_folder, new_folder_name))

if __name__ == '__main__':
    # 사용 예시
    parent_folder_path = "/home/work/main/jpark/Event_camera/data/Town_train"  # 기준 폴더 경로를 지정합니다.
    rename(parent_folder_path)