import os

def count_second_level_subdirectories(directory):
    try:

        first_level_entries = os.listdir(directory)
        result = {}
        
        for entry in first_level_entries:
            first_level_path = os.path.join(directory, entry)
            # print(first_level_path,first_level_path[-4:])
            if first_level_path[-4:] == '_old' or first_level_path[-4:] == 'test':
                continue
            if os.path.isdir(first_level_path):

                second_level_entries = os.listdir(first_level_path)

                second_level_count = 0
                for sub_entry in second_level_entries:
                    sub_path = os.path.join(first_level_path, sub_entry)
                    if os.path.isdir(sub_path):
                        second_level_count += 1
                result[entry] = second_level_count

        total_count = sum(result.values())
        
        return result, total_count
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return {}, -1
    except PermissionError:
        print(f"Permission denied to access the directory {directory}.")
        return {}, -1


directory_path = '/data2/chaonan/cloth_traj_data/'
second_level_counts, total_count = count_second_level_subdirectories(directory_path)

total_cloth = 0
total_tops = 0
total_dress = 0
total_skirt = 0
total_pants = 0

for first_level_dir, count in second_level_counts.items():
    if first_level_dir in ['DLG', 'DLNS', 'DLT', 'DSNS', 'SL', 'SS', 'TCNC', 'TCNO', 'THNC', 'TNNC']:
        cloth_num = count
    elif first_level_dir in ['DSSS', 'DLSS', 'TCSC', 'TNSC', 'DSLS', 'DLLS', 'TNLC', 'TNLO', 'TCLC', 'TCLO', 'THLO', 'THLC']:
        cloth_num = count / 3
    elif first_level_dir in ['PL', 'PS']:
        cloth_num = count / 2
    else:
        assert 0

    print(f"'{first_level_dir}' has {count} second-level subdirectories, {cloth_num} clothes.")
    total_cloth += cloth_num
        
    if first_level_dir[0] == 'T':
        total_tops += cloth_num
    elif first_level_dir[0] == 'D':
        total_dress += cloth_num
    elif first_level_dir[0] == 'S':
        total_skirt += cloth_num
    elif first_level_dir[0] == 'P':
        total_pants += cloth_num

assert total_tops+total_dress+total_skirt+total_pants == total_cloth
# 打印总计
print(f"Total number of second-level subdirectories: {total_count}. Total Cloth: {total_cloth}")
print(f'Tops: {total_tops}, Dress: {total_dress}, Skirt: {total_skirt}, Pants: {total_pants}')