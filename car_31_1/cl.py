from glob import glob
import os

ds =glob('*.txt')

for d in ds:
    if 'classes.txt' in d:
        continue
    f = open(d, 'r')
    ls = f.readlines()
    f.close()
    new_label_str = ''
    pcnt = 0
    for l in ls:
        cx = int(1920 * float(l.split(' ')[1]))
        cy = int(1080 * float(l.split(' ')[2]))
        
        cls = int(l.split(' ')[0])
        if cls == 2:
            new_label_str += l
            #new_label_str += ('2 ' + ' '.join(l.split(' ')[1:]))
            break

    f = open(d, 'w')
    f.write(new_label_str)
    f.close()
