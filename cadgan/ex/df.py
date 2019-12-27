"""
Module containig utility functions for processing/display results (tensorboard
logs) saved as dataframes.
"""
import base64
import io

import numpy as np
import pandas as pd
import PIL
import skimage


def image_base64(im, resize=224):
    """
    im: a numpy array whose range is [0,1]
    Return a base64 encoding of an image.
    """
    with io.BytesIO() as buffer:
        # resize
        im = skimage.transform.resize(im, (resize, resize), mode="reflect", anti_aliasing=True)

        # convert the numpy array to a PIL image
        pil_im = PIL.Image.fromarray(np.uint8(im * 255))
        pil_im.save(buffer, "jpeg")
        return base64.b64encode(buffer.getvalue()).decode()


def images_formatter(imgs, col=2, width=96):
    """
    This function is used to format images in a pandas dataframe (shown in a
    table).

    width: width of each image to display (in pixels).
    """
    # https://www.kaggle.com/stassl/displaying-inline-images-in-pandas-dataframe
    if len(imgs) < col:
        html = u'<div class="df">'
    else:
        html = u'<div class="df" style="width: {}px">'.format(col * width + 15)

    for i in range(len(imgs)):
        # html += '<div class="imgs" style="display: inline-block; width: {}px">'.format(width+30)
        html += '<div class="imgs" style="display: inline-block; max-width: {}px">'.format(width + 10)
        img64 = image_base64(imgs[i], resize=width)
        #         print(img64)
        img_tag = '<img src="data:image/jpeg;base64,{}">'.format(img64)
        #         img_tag = f'<img src="data:image/jpeg;base64,{img64}">'
        html += img_tag + " </div> "

        if (i + 1) % col == 0:
            html += "<br>"
    #     print(html)
    return html + "</div>"


def df_to_html(df, img_formatter=images_formatter):
    """
    Convert a dataframe (containing Tenboard results) into HTML for display.
    Each row in the dataframe is one run of cadgan.ex.run_lars_gkmm.py
    """
    pd.set_option("display.max_colwidth", -1)
    pd.set_option("display.max_columns", -1)
    cond_formatter = lambda imgs: images_formatter(imgs, col=1)
    html_table = df.to_html(
        formatters={
            "cond_imgs": cond_formatter,
            "out_imgs": img_formatter,
            "feat_imgs": img_formatter,
            "cond_feat": img_formatter,
            "ini_imgs": img_formatter,
        },
        escape=False,
        border=0,
    )
    html = """
    <html>
    <style>
    td{{
        border: 1px solid #444444;
        padding: 5px;
    }}
    table {{ 
        border-spacing: 0px;
        border-collapse: separate;
    }}
    tr:nth-child(even) {{
        background: #f2f2f2; 
    }}
        
    </style>

    <body>
    {}

    <br><br> <br><br> <br><br> <br><br> <br><br> 
    </body>
    </html>
    """.format(
        html_table
    )
    return html
