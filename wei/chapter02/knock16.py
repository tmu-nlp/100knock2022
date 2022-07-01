def n_seg(file, n):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        total_rows = int(len(data))
        remainder = total_rows % n
        seg_rows = total_rows // n   # rows num in each part
        cnt = 0
        for i in range(n):
            with open(f'knock16_out_{i}.txt', 'w', encoding='utf-8') as f_out:
                for _ in range(seg_rows):
                    f_out.write(data[cnt])
                    cnt += 1
                if remainder != 0:
                    f_out.write(data[cnt])
                    cnt += 1
                    remainder -= 1


if __name__ == '__main__':
    file = 'popular-names.txt'
    n_seg(file, 11)









