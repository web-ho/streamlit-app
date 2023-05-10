This is an ongoing and learning project. There might be lots of inefficient and poor code. Feedbacks are welcome. 

You can use the app here- 
<a href="https://rider110-know-the-bird.streamlit.app/" target="_blank">
    <button style="background-color: #008CBA; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
        Know-The-Bird
    </button>
</a>

This project classifies 25 Indian Bird Species found throughout India. For now I am only classifying images, later we will extend this project to work in real-time and also give important information related to the species. And other cool facts.

I have used accuracy as a metric to track and improve model performance, which I have realised should not be your ideal choice. Rather, use precision-recall.

The dataset used for this project can be found at-
<a href="https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images" target="_blank">
    <button style="background-color: #008CBA; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
        Dataset Here
    </button>
</a>

Due to LFS bandwidth limit. I needed to reduce model size, which may have resulted in a decrease in accuracy. 
I used pytorch's method of model quantization. This has to be applied post-training. There is other method of pruning too, it is little complex
to apply. 



I have some ideas to extend this project beyond classification. 

- First, to gather and add more species.
- Introduce a pipeline to label images with bboxes.
- Use the SAM model for bird segmentation.
- Also, use OpenAI's API to get info about the recognised bird species.
