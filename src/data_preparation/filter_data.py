import os
from PIL import Image
import numpy as np
#import tensorflow as tf
import pandas as pd
#import difPy

WIDTH = 650
HEIGHT = 450


def move_files(name: str):
    # filter_isic2019_train(start=15000, end=None, name='15k_to_end')
    data_dir = '/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data'
    meta = pd.read_csv(f"{data_dir}/metadata/{name}.csv")

    for _, row in meta.iterrows():
        image_id = row['image_id']
        label = row['dx']
        quality = row['dx_type']
        if 'histo' not in quality:
            continue
        if type(label) is not str:
            continue
        if 'nevus' in label or 'nv' in label:
            os.rename(f'{data_dir}/UDA/UDA1/600x450/{image_id}.png', f'{data_dir}/UDA/UDA/nv/{image_id}.png')
        if 'melanosis' in label:
            continue
        if 'mel' in label:
            os.rename(f'{data_dir}/UDA/UDA1/600x450/{image_id}.png', f'{data_dir}/UDA/UDA/mel/{image_id}.png')


def move_files_dir():
    dataset_dir = '/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data/VIENNA/ISIC2020_VIENNA1/600x450'
    data_dir = '/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data'
    meta = pd.read_csv(f"{data_dir}/metadata/processed_UDA2.csv")
    files = [file for file in os.listdir(dataset_dir) if file.endswith('.png')]

    for image_id in files:
        label = meta[meta['image_id'] == image_id[:-4]]['dx'].item()
        if type(label) is not str:
            continue
        if 'nevus' in label or 'nv' in label:
            os.rename(f'{data_dir}/UDA/UDA2/600x450/{image_id}', f'{data_dir}/UDA/UDA/nv/{image_id}')
        if 'mel' in label:
            os.rename(f'{data_dir}/UDA/UDA2/600x450/{image_id}', f'{data_dir}/UDA/UDA/mel/{image_id}')


def cat_meta(name: str):
    df = pd.DataFrame()
    dir = "/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data/metadata"
    for file in os.listdir(dir):
        if name in file:
            file_path = os.path.join(dir, file)
            part = pd.read_csv(file_path)
            df = pd.concat([df, part])
    df.to_csv(os.path.join(dir, name))


def filter_dy(name, delete=False):
    data_dir = f'/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data/{name}'
    meta = pd.read_csv(f"{data_dir}/../metadata/{name}.csv")
    files = [(file, os.path.join(d, file)) for d,_,f in os.walk(data_dir) for file in f if file.endswith('.png')]

    not_histo = list()
    histo = list()

    for image_id, path in files:
        dy = meta[meta['image_id'] == image_id[:-4]]['dx_type'].item()
        if isinstance(dy, str):
            if 'histo' in dy:
                histo.append(path)
        if not dy == dy:
            not_histo.append(path)
        elif 'hist' not in dy:
            not_histo.append(path)

    print(len(not_histo))
    print(len(histo))

    if delete:
        for file in not_histo:
            os.remove(file)


def search_duplicates(dir, delete: bool = False, similarity='duplicates'):
    data_dir = '/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data/'
    data_dir += dir
    dif = difPy.build(data_dir, recursive=True, show_progress=True)
    search = difPy.search(dif, similarity=similarity)
    print(search)
    if delete:
        search.delete(silent_del=False)


def preparebcn():
    import os
    from PIL import Image
    import pandas as pd
    # metadata
    meta = pd.read_csv('../../datasets/metadata/bcn1920.csv')
    # filter
    meta_fil = meta[(meta['diagnosis'].isin(['melanoma', 'nevus'])) & (meta['diagnosis_confirm_type'] == 'histopathology')]
    #reset index format
    meta_fil = meta_fil.reset_index().drop(columns=['index'])
    # delete .png
    dir20 = [x[:-4] for x in os.listdir('../../datasets/ISIC-images1920')]
    # list to save not found image ids
    #missed = list()
    # go through image ids
    for img_id in dir20:
        # check if the img id is in metadata
        if img_id in meta_fil['isic_id'].values:
            # Open the image
            img = Image.open('../../datasets/ISIC-images1920/' + img_id + '.JPG')
            # Reformat the image
            img = img.resize((600, 450))
            img.save(
                '../../datasets/isic1920_fil/' + str(meta_fil[meta_fil['isic_id'] == img_id]['diagnosis'].values[0]) + '/' + img_id + '.png')
        #else:
         #   missed.append(img_id)
    #print(missed)


if __name__ == "__main__":
    preparebcn()
    #filter_dy('BCN', delete=True)
    exit()
    #filter_isic2019_train(start=15000, end=None, name='15k_to_end')
    #dataset_dir = '/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data/DERM7PT/600x450'
    #data_dir = '/Users/floyd/Documents/Studium/DS/FreiesProjekt/Self Supervised Fine Tuning/data'
    #meta = pd.read_csv(f"{data_dir}/metadata/processed_DERM7PT.csv")
    #files = [file for file in os.listdir(dataset_dir) if file.endswith('.png')]

    #for image_id in files:
     #   label = meta[meta['image_id'] == image_id[:-4]]['dx'].item()
      #  if type(label) is not str:
       #     continue
        #if 'nevus' in label or 'nv' in label:
        #    os.rename(f'{data_dir}/DERM7PT/600x450/{image_id}', f'{data_dir}/DERM7PT/nv/{image_id}')
        #if 'mel' in label:
         #   os.rename(f'{data_dir}/DERM7PT/600x450/{image_id}', f'{data_dir}/DERM7PT/mel/{image_id}')

    # for _, row in meta.iterrows():
    #     image_id = row['image_id']
    #     label = row['dx']
    #     if type(label) is not str:
    #         continue
    #     if 'nevus' in label or 'nv' in label:
    #         os.rename(f'{data_dir}/UDA/UDA1/600x450/{image_id}.png', f'{data_dir}/UDA/UDA/nv/{image_id}.png')
    #     if 'mel' in label:
    #         os.rename(f'{data_dir}/UDA/UDA1/600x450/{image_id}.png', f'{data_dir}/UDA/UDA/mel/{image_id}.png')
