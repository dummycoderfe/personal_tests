
print("shape, norm_shape, ck_dgrad, ck_wgrad, torch_wgrad, torch_dgrad, apex_wgrad, apex_dgrad")

with open('ln_bw.csv', 'r') as f:

    lines = f.read()
    v = ["" for _ in range(7)]
    cnt = 1
    for d in lines.splitlines():
        if 'shape' in d:
            cnt = 1
            x = d[6:]
            x = x.replace(', ', 'x')
            x = x.replace('] [', ', ')
            x = x.replace(']', '')
            x = x.replace('[', '')
            v[0] = x
        else:
            x = d.split(',')[1]
            v[cnt] = x
            cnt += 1
        if cnt == 7:
            print(','.join(v))
