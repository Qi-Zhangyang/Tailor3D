# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from PIL import Image
import numpy as np
import gradio as gr


def assert_input_image(input_front_image, input_back_image):
    if input_front_image is None:
        raise gr.Error("No front image selected or uploaded!")
    if input_back_image is None:
        raise gr.Error("No back image selected or uploaded!")

def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def init_preprocessor():
    from openlrm.utils.preprocess import Preprocessor
    global preprocessor
    preprocessor = Preprocessor()

def preprocess_fn(image_in_front: np.ndarray, image_in_back: np.ndarray, remove_bg: bool, recenter: bool, working_dir):
    # save front image first
    image_raw_front = os.path.join(working_dir.name, "raw_front.png")
    with Image.fromarray(image_in_front) as img:
        img.save(image_raw_front)
    image_out_front = os.path.join(working_dir.name, "front/rembg_front.png")

    # save back image first
    image_raw_back = os.path.join(working_dir.name, "raw_back.png")
    with Image.fromarray(image_in_back) as img:
        img.save(image_raw_back)
    image_out_back = os.path.join(working_dir.name, "back/rembg_back.png")

    # process the front and back image.
    success_front = preprocessor.preprocess(image_path=image_raw_front, save_path=image_out_front, rmbg=remove_bg, recenter=recenter)
    success_back = preprocessor.preprocess(image_path=image_raw_back, save_path=image_out_back, rmbg=remove_bg, recenter=recenter)
    assert success_front and success_back, f"Failed under preprocess_fn!"
    return image_out_front, image_out_back


def demo_openlrm(infer_impl):

    def core_fn(image_front: str, image_back: str, source_cam_dist: float, working_dir):
        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_mesh_path = os.path.join(working_dir.name, "output.ply")
        infer_impl(
            image_path=image_front,
            source_cam_dist=source_cam_dist,
            export_video=True,
            export_mesh=False,
            dump_video_path=dump_video_path,
            dump_mesh_path=dump_mesh_path,
            image_path_back=image_back,
        )
        return dump_video_path


    _TITLE = '''🔥 🔥 🔥 Tailor3D: Customized 3D Assets Editing and Generation with Dual-Side Images'''

    _DESCRIPTION = '''
        <div>
            <a style="display:inline-block" href='https://github.com/Qi-Zhangyang/Tailor3D'><img src='https://img.shields.io/github/stars/Qi-Zhangyang/Tailor3D?style=social'/></a>
            <a style="display:inline-block; margin-left: .5em" href="https://huggingface.co/alexzyqi"><img src='https://img.shields.io/badge/Model-Weights-blue'/></a>
        </div>
        We propose Tailor3D, a novel pipeline creating customized 3D assets from editable dual-side images and feed-forward reconstruction methods.

        Here we show the final step of Tailor3D. That is given the edited front and beck view of the object. We can produce the 3D object with several seconds.

        <strong>Disclaimer:</strong> This demo uses `Tailor3D-base-1.1` model with 288x288 rendering resolution here for a quick demonstration.
    '''

    with gr.Blocks(analytics_enabled=False) as demo:

        # HEADERS
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        with gr.Row():
            gr.Markdown(_DESCRIPTION)

        # DISPLAY
        with gr.Row():
            gr.Markdown(
                """
                ## 🖼️ Input: This is the input front and back images.
                """
            )
        with gr.Row():
            with gr.Column(variant='panel', scale=0.2):
                with gr.Tabs(elem_id="tailor3d_input_front_image"):
                    with gr.TabItem('Input Front-view Image'):
                        with gr.Row():
                            input_front_image = gr.Image(label="Input Front Image", image_mode="RGBA", width="auto", sources="upload", type="numpy", elem_id="content_image")

            with gr.Column(variant='panel', scale=0.2):
                with gr.Tabs(elem_id="tailor3d_input_back_image"):
                    with gr.TabItem('Input Back-view Image'):
                        with gr.Row():
                            input_back_image = gr.Image(label="Input Back Image", image_mode="RGBA", width="auto", sources="upload", type="numpy", elem_id="content_image")
        with gr.Row():
            gr.Markdown(
                """
                ## 🛠️ Preprocess: Remove the background and center the object.
                """
            )
        with gr.Row():
            with gr.Column(variant='panel', scale=0.2):
                with gr.Tabs(elem_id="tailor3d_processed_image"):
                    with gr.TabItem('Processed Front-view Image'):
                        with gr.Row():
                            processed_front_image = gr.Image(label="Processed Image", image_mode="RGBA", type="filepath", elem_id="processed_image", width="auto", interactive=False)
            with gr.Column(variant='panel', scale=0.2):
                with gr.Tabs(elem_id="tailor3d_processed_image"):
                    with gr.TabItem('Processed Back-view Image'):
                        with gr.Row():
                            processed_back_image = gr.Image(label="Processed Image", image_mode="RGBA", type="filepath", elem_id="processed_image", width="auto", interactive=False)
        with gr.Row():
            gr.Markdown(
                """
                ## 🚀 Output: The rendering video of the 3D object.
                Note that the output is the 3D mesh, for convience, we showcase it through a video that circles around.
                """
            )
        with gr.Row():
            with gr.Column(variant='panel', scale=0.2):
                with gr.Tabs(elem_id="tailor3d_render_video"):
                    with gr.TabItem('Rendered Video'):
                        with gr.Row():
                            output_video = gr.Video(label="Rendered Video", format="mp4", width="auto", autoplay=True)

        # SETTING
        with gr.Row():
            with gr.Column(variant='panel', scale=1):
                with gr.Tabs(elem_id="openlrm_attrs"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            gr.Markdown(
                                """
                                <strong>Best Practice</strong>:
                                    Centered objects in reasonable sizes. Try adjusting source camera distances.
                                """
                            )
                            checkbox_rembg = gr.Checkbox(True, label='Remove background')
                            checkbox_recenter = gr.Checkbox(True, label='Recenter the object')
                            slider_cam_dist = gr.Slider(1.0, 3.5, value=2.0, step=0.1, label="Source Camera Distance")
                            submit = gr.Button('Generate', elem_id="openlrm_generate", variant='primary')

        # EXAMPLES
        with gr.Row():
            gr.Markdown(
                """
                ## Example in the paper.
                ### A. 3D Style Transfer
                Here we keep the object ID and just transfer the style. <br>

                **Line 1: A pop-mart boy with astronaut, blue, traditional Chinese and grey style.**
                """
            )
        with gr.Row():
            examples = [
                ['assets/sample_input/demo/front/boy_astronaut.png', 'assets/sample_input/demo/back/boy_astronaut.png'],
                ['assets/sample_input/demo/front/boy_blue.png', 'assets/sample_input/demo/back/boy_blue.png'],
                ['assets/sample_input/demo/front/boy_chinese_style.png', 'assets/sample_input/demo/back/boy_chinese_style.png'],
                ['assets/sample_input/demo/front/boy_grey_clothes.png', 'assets/sample_input/demo/back/boy_grey_clothes.png'],
            ]

            for example in examples:
                with gr.Column(scale=0.3):
                    gr.Examples(
                        examples=[example],
                        inputs=[input_front_image, input_back_image], 
                        outputs=None, # [processed_image, output_video],
                        fn=None, # example_fn,
                        cache_examples=bool(os.getenv('SPACE_ID')),
                        examples_per_page=3,
                    )

        # EXAMPLES
        with gr.Row():
            gr.Markdown(
                """
                **Line 2: A LEGO model featuring an astronaut, green and red elements, and a wizard theme.**
                """
            )
        with gr.Row():
            examples = [
                ['assets/sample_input/demo/front/lego_astronaut.png', 'assets/sample_input/demo/back/lego_astronaut.png'],
                ['assets/sample_input/demo/front/lego_green.png', 'assets/sample_input/demo/back/lego_green.png'],
                ['assets/sample_input/demo/front/lego_red.png', 'assets/sample_input/demo/front/lego_red.png'],
                ['assets/sample_input/demo/front/lego_wizard.png', 'assets/sample_input/demo/back/lego_wizard.png'],
            ]

            for example in examples:
                with gr.Column(scale=0.3):
                    gr.Examples(
                        examples=[example],
                        inputs=[input_front_image, input_back_image], 
                        outputs=None, # [processed_image, output_video],
                        fn=None, # example_fn,
                        cache_examples=bool(os.getenv('SPACE_ID')),
                        examples_per_page=3,
                    )
        with gr.Row():
            gr.Markdown(
                """
                **Line 3: A marvel boy featuring an Captain America, Ironman and Spiderman, and a Superman theme.**
                """
            )
        with gr.Row():
            examples = [
                ['assets/sample_input/demo/front/marvel_captain.png', 'assets/sample_input/demo/back/marvel_captain.png'],
                ['assets/sample_input/demo/front/marvel_ironman.png', 'assets/sample_input/demo/front/marvel_ironman.png'],
                ['assets/sample_input/demo/front/marvel_spiderman.png', 'assets/sample_input/demo/back/marvel_spiderman.png'],
                ['assets/sample_input/demo/front/marvel_superman.png', 'assets/sample_input/demo/back/marvel_superman.png'],
            ]

            for example in examples:
                with gr.Column(scale=0.3):
                    gr.Examples(
                        examples=[example],
                        inputs=[input_front_image, input_back_image], 
                        outputs=None, # [processed_image, output_video],
                        fn=None, # example_fn,
                        cache_examples=bool(os.getenv('SPACE_ID')),
                        examples_per_page=3,
                    )
        # EXAMPLES
        with gr.Row():
            gr.Markdown(
                """
                ### B. 3D Generative Geometry or Pattern Fill
                
                Here, we start with a simple object and gradually add various accessories, costumes, or patterns step by step. We only showcase the final effect after multiple rounds of decoration. <br>

                **Line 4: Initial object: sofa, dog, penguin, house.**
                """
            )
        with gr.Row():
            examples = [
                ['assets/sample_input/demo/front/sofa.png', 'assets/sample_input/demo/back/sofa.png'],
                ['assets/sample_input/demo/front/space_dog.png', 'assets/sample_input/demo/back/space_dog.png'],
                ['assets/sample_input/demo/front/penguin.png', 'assets/sample_input/demo/back/penguin.png'],
                ['assets/sample_input/demo/front/house.png', 'assets/sample_input/demo/back/house.png'],
            ]

            for example in examples:
                with gr.Column(scale=0.3):
                    gr.Examples(
                        examples=[example],
                        inputs=[input_front_image, input_back_image], 
                        outputs=None, # [processed_image, output_video],
                        fn=None, # example_fn,
                        cache_examples=bool(os.getenv('SPACE_ID')),
                        examples_per_page=3,
                    )

        with gr.Row():
            gr.Markdown(
                """
                ### C. 3D Style Fusion
                
                We will maintain a consistent front style of the object while continuously changing the back style, blending the two different styles into one object.<br>

                **Line 5: A bird with different back styles.**
                """
            )
        with gr.Row():
            examples = [
                ['assets/sample_input/demo/front/bird.png', 'assets/sample_input/demo/back/bird.png'],
                ['assets/sample_input/demo/front/bird_brownblue.png', 'assets/sample_input/demo/back/bird_brownblue.png'],
                ['assets/sample_input/demo/front/bird_rainbow.png', 'assets/sample_input/demo/back/bird_rainbow.png'],
                ['assets/sample_input/demo/front/bird_whitered.png', 'assets/sample_input/demo/back/bird_whitered.png'],
            ]

            for example in examples:
                with gr.Column(scale=0.3):
                    gr.Examples(
                        examples=[example],
                        inputs=[input_front_image, input_back_image], 
                        outputs=None, # [processed_image, output_video],
                        fn=None, # example_fn,
                        cache_examples=bool(os.getenv('SPACE_ID')),
                        examples_per_page=3,
                    )

        with gr.Row():
            gr.Markdown(
                """
                ### Others
                I vote for kunkun forever, I am really an I-kUN and have heard many of his songs.
                """
            )
        with gr.Row():
            examples = [
                ['assets/sample_input/demo/front/loopy.png', 'assets/sample_input/demo/back/loopy.png'],
                ['assets/sample_input/demo/front/mario.png', 'assets/sample_input/demo/back/mario.png'],
                ['assets/sample_input/demo/front/armor.png', 'assets/sample_input/demo/back/armor.png'],
                ['assets/sample_input/demo/front/kunkun_law.png', 'assets/sample_input/demo/back/kunkun_law.png'],
            ]

            for example in examples:
                with gr.Column(scale=0.3):
                    gr.Examples(
                        examples=[example],
                        inputs=[input_front_image, input_back_image], 
                        outputs=None, # [processed_image, output_video],
                        fn=None, # example_fn,
                        cache_examples=bool(os.getenv('SPACE_ID')),
                        examples_per_page=3,
                    )

        working_dir = gr.State()
        submit.click(
            fn=assert_input_image,
            inputs=[input_front_image, input_back_image],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=preprocess_fn,
            inputs=[input_front_image, input_back_image, checkbox_rembg, checkbox_recenter, working_dir],
            outputs=[processed_front_image, processed_back_image],
        ).success(
            fn=core_fn,
            inputs=[processed_front_image, processed_back_image, slider_cam_dist, working_dir],
            outputs=[output_video],
        )

        demo.queue()
        demo.launch()


def launch_gradio_app():

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "alexzyqi/Tailor3D-Base-1.0",
        "APP_PRETRAIN_MODEL_NAME": "zxhezexin/openlrm-mix-base-1.1",
        "APP_INFER": "./configs/infer-gradio.yaml",
        "APP_TYPE": "infer.lrm",
        "NUMBA_THREADING_LAYER": 'omp',
    })

    from openlrm.runners import REGISTRY_RUNNERS
    from openlrm.runners.infer.base_inferrer import Inferrer
    InferrerClass : Inferrer = REGISTRY_RUNNERS[os.getenv("APP_TYPE")]
    with InferrerClass() as inferrer:
        init_preprocessor()
        if not bool(os.getenv('SPACE_ID')):
            from openlrm.utils.proxy import no_proxy
            demo = no_proxy(demo_openlrm)
        else:
            demo = demo_openlrm
        demo(infer_impl=inferrer.infer_single)


if __name__ == '__main__':

    launch_gradio_app()
