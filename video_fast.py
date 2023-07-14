
from fastapi import FastAPI
import gradio as gr
import os
import uvicorn
from face_detection import select_face
from face_swap import face_swap

app = FastAPI()
@app.get('/test')
def test():
    return ('testapp, hello, how are you?')


import os
import cv2
import logging
import argparse

from face_detection import select_face
from face_swap import face_swap

def get_result(inputs_video, inputs_image):
    src_points, src_shape, src_face = select_face(cv2.imread(inputs_image))
    if src_points is None:
        print('No face detected in the source image !!!')
        exit(-1)
    print('123')
    save_path = 'result/result.mp4'
    video = cv2.VideoCapture(inputs_video)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), video.get(cv2.CAP_PROP_FPS),
                                      (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while video.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, dst_img = video.read()
        dst_points, dst_shape, dst_face = select_face(dst_img, choose=False)
        if dst_points is not None:
            dst_img = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, 68)
        writer.write(dst_img)

    video.release()
    writer.release()    

    return writer

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Face Swap")
    with gr.Row():
        with gr.Column():
            inputs_video = gr.Video(label="Original Video", show_label=True)
            inputs_image = gr.Image(source='upload')
            with gr.Row():
                runBtn = gr.Button("FaceSwap")
        with gr.Column():
            # gallery = gr.Gallery(label="Generated images", show_label=False)
            result = gr.Video(label="Generated Video", show_label=True)
    runBtn.click(fn=get_result, inputs=[inputs_video, inputs_image], outputs=[result])

gr.mount_gradio_app(app, block, path='/')
if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0" )

