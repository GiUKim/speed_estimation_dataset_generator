from glob import glob
import os
import numpy as np
import shutil
from tqdm import tqdm
import cv2
import sys
import random
import math

def mk(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def mkdir_split():
    mk(os.path.join(dataset_save_path, 'train', 'phase1'))
    mk(os.path.join(dataset_save_path, 'validation'))
    mk(os.path.join(cache_path, 'phase1'))

def get_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

dataset_save_path = 'dataset_eval'
cache_path = 'cache_eval'
dirs = glob('car_*_*')
cls = 1 if 'car' in dirs[0] else 0
limit_max_speed, limit_min_speed = 200.0, 0.0
base_itercount = 60
imw, imh = 960,540

dds = [sorted(glob(dir + '/*.txt'), key=lambda x : int(x.split('/')[-1].split('_')[-1].split('.')[0])) for dir in dirs]

lins = np.linspace(0.0, 15.0, 30).tolist()
speed_lins = np.linspace(limit_min_speed, limit_max_speed, 20).tolist()
print(lins)
print(speed_lins)
mklist = []
phase_idx = 1
push_count = 0
dup_count = 0

for time_index in range(0, len(lins)-1):
    #if lins[time_index] < 4.6:
    #    continue
    for speed_index in range(0, len(speed_lins)-1):
        #if lins[time_index] < 4.6 and speed_lins[speed_index] < 21.0:
        #    continue
        local_dup_count = 0
        pass_count = 0
        push_count = 0
        progress = (time_index) * (len(speed_lins) - 1) + speed_index+1
        total_progress = (len(lins)-1)*(len(speed_lins)-1)
        print(f'[{progress}/{total_progress}] time range: {round(lins[time_index], 2)} sec ~ {round(lins[time_index+1], 2)} sec, speed range: {round(speed_lins[speed_index], 2)} km/h ~ {round(speed_lins[speed_index+1], 2)} km/h')
        while True:
            if pass_count > 500:
                print('break by passcount')
                break
            rdirs = random.choice(dds)
            #min_speed = g_min_speed
            min_speed = speed_lins[speed_index]
            max_speed = speed_lins[speed_index+1]

            base_speed = float(rdirs[0].split('/')[-2].split('_')[1])

            frame_lower_bound = 0
            frame_upper_bound = len(rdirs) - 1

            r1 = random.randint(frame_lower_bound, frame_upper_bound)
            r2 = random.randint(frame_lower_bound, frame_upper_bound)
            if r1 < r2:
                end_index = r2
                random_start_index = r1
            else:
                end_index = r1
                random_start_index = r2
            n = end_index - random_start_index + 1
            
            alpha_min = lins[time_index]
            alpha_max = lins[time_index+1]
                
            alpha = random.uniform(alpha_min, alpha_max)
            tm = alpha * 1.0  #sec

            new_speed = base_speed * (n / (60.0 * alpha))
            if (new_speed > max_speed or new_speed < min_speed) :
                pass_count += 1
                continue

            
            sp = rdirs[random_start_index]
            ep = rdirs[end_index]
            
            f_start = open(sp, 'r')
            l_start = f_start.readlines()[0]
            f_start.close()

            f_end = open(ep, 'r')
            l_end = f_end.readlines()[0]
            f_end.close()
            
            sinfo = [float(f) for f in l_start.split(' ')[1:]]
            start_x1 = int(imw * (sinfo[0] - (sinfo[2] / 2.0)))
            start_y1 = int(imh * (sinfo[1] - (sinfo[3] / 2.0)))
            start_x2 = int(imw * (sinfo[0] + (sinfo[2] / 2.0)))
            start_y2 = int(imh * (sinfo[1] + (sinfo[3] / 2.0)))
            einfo = [float(f) for f in l_end.split(' ')[1:]]
            end_x1 = int(imw * (einfo[0] - (einfo[2] / 2.0)))
            end_y1 = int(imh * (einfo[1] - (einfo[3] / 2.0)))
            end_x2 = int(imw * (einfo[0] + (einfo[2] / 2.0)))
            end_y2 = int(imh * (einfo[1] + (einfo[3] / 2.0)))
            tm_int = int(tm)
            if get_iou(sinfo, einfo) > 0.7:
                pass_count += 1
                continue
            cont_flag = False
            ifs = f"{start_x1}_{start_y1}_{start_x2}_{start_y2}_{end_x1}_{end_y1}_{end_x2}_{end_y2}_{cls}_{tm_int}"
            for it in (1, phase_idx + 30):
                if (os.path.exists(os.path.join(cache_path, f'phase{it}', ifs))):
                    #print('duplicate find')
                    dup_count += 1
                    pass_count += 1
                    local_dup_count += 1
                    cont_flag = True
                    break
            if cont_flag:
                continue
            valid = True
            if (start_x1 < 0) or (start_y1 < 0) or (start_x2 >= imw) or (start_y2 >= imh):
                valid = False
            if (end_x1 < 0) or (end_y1 < 0) or (end_x2 >= imw) or (end_y2 >= imh):
                valid = False
            if not valid:
                #print('unvalid!!!!!!!!!!!!!!!!!!!!')
                continue
            push_count += 1
            pass_count = 0
            f = open(os.path.join(cache_path, f'phase{phase_idx}', ifs), 'w')
            f.close()
            #cv2.rectangle(bg, (start_x1, start_y1), (start_x2, start_y2), (0, 255, 0), 1)
            #cv2.rectangle(bg, (end_x1, end_y1), (end_x2, end_y2), (0, 255, 0), 1)
            #cv2.putText(bg, f"{round(tm, 2)} sec,  {round(new_speed, 2)} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.imwrite(os.path.join('sample', f'{start_x1}{start_y1}{start_x2}{start_y2}_{end_x1}{end_y1}{end_x2}{end_y2}_{tm}_{new_speed}.jpg'), bg)
            #cv2.imshow('bg', bg)
            #k  =cv2.waitKey(0)
            #if k == ord('q'):
            #    sys.exit(1)
            new_path = ''
            new_path += l_start.replace('\n', '_').replace(' ', '_')
            new_path +=  (l_end.replace('\n', '_')).replace(' ', '_')
            new_path += f"_{tm}_{new_speed}"
            lst = [float(f) for f in new_path.replace('__', '_').split('_')]
            n = ''
            for idx, l in enumerate(lst):
                if idx in [0, 5]:
                    n += "1_"
                elif idx in [1, 2, 3, 4, 6, 7, 8, 9, 10]:
                    n += f"{round(l, 6)}_"
                elif idx == 11:
                    n += f"{round(l, 2)}.txt"
            new_path = n

            #print('-----------------')
            #print(l_start)
            #print(l_end)
            #print(f'time  : {round(tm, 2)} sec')
            #print(f'speed : {round(new_speed, 2)} km/h')
            #print('-----------------')

            tv = random.randint(1, 60)
            if tv == 1:
                f = open(os.path.join(dataset_save_path, 'validation', new_path), 'w')
                f.close()
            else:
                try:
                    f = open(os.path.join(dataset_save_path, 'train', f'phase{phase_idx}', new_path), 'w')
                    f.close()
                except:
                    phase_idx += 1
                    try:
                        os.makedirs(os.path.join(dataset_save_path, 'train', f'phase{phase_idx}'))
                        os.makedirs(os.path.join(cache_path, f'phase{phase_idx}'))
                    except:
                        pass
                    for _ in range(0, 10):
                        print('no space in directory, increse phase_idx: ', phase_idx)
            if push_count == 100:
                print('Local dup count: ', local_dup_count)
                break

print('push: ', push_count)
print('duplicate: ', dup_count)
