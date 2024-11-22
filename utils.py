import os

def save_log(out_dir, date, **kwargs):

    path = os.path.join(out_dir, 'log.txt')
    with open(path , "w") as f:
        f.write(f"Date/time of creation : {date}\n")
        for k, v in kwargs.items():
            if isinstance(v, list):
                if len(v) == 0: continue
            f.write(f"{k} : {v}\n")
