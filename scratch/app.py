
import gradio as gr
import requests
import os

# SeehackAIthonWarmup1_0.ipynb for discussion when to use Authorization
# bearertoken = "" #os.environ['HACKAITHONBEARERTOKEN']  
headers = {"Authorization": "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" }

def query(filepath):
    print (filepath);
    data = open(filepath, 'rb' ).read()
    from huggingface_hub import from_pretrained_fastai
    learner = from_pretrained_fastai("kenlomax/zx80zx81a") 
    _,_,probs = learner.predict(PILImage.create(data))
    return ( learner.dls.vocab , ": ", probs)

def useMyModel(image):
    output = query( image)
    print (str(output))
    return str(output)

iface = gr.Interface(
  fn=useMyModel, inputs=[gr.Image(type="filepath")], outputs="text")
iface.launch()
