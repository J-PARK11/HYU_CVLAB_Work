import os

def add_suffix_to_folders(parent_folder, suffix):
    """
    폴더 안에 있는 모든 폴더의 이름에 접미사를 추가하는 함수

    Parameters:
        parent_folder (str): 기준 폴더 경로
        suffix (str): 추가할 접미사

    Returns:
        None
    """
    # 기준 폴더 안에 있는 모든 폴더 목록 가져오기
    folder_list = os.listdir(parent_folder)

    # 모든 폴더에 접미사를 추가하여 이름을 변경
    for folder in folder_list:
        new_folder_name = suffix + '_' + folder
        os.rename(os.path.join(parent_folder, folder), os.path.join(parent_folder, new_folder_name))

if __name__ == '__main__':
    # 사용 예시
    parent_folder_path = "/home/work/main/jpark/Event_camera/data/Town05_test"  # 기준 폴더 경로를 지정합니다.
    suffix_to_add = "Town05"  # 추가할 접미사를 지정합니다.
    add_suffix_to_folders(parent_folder_path, suffix_to_add)