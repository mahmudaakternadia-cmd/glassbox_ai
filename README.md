🔍 𝗚𝗹𝗮𝘀𝘀𝗕𝗼𝘅 𝗔𝗜:

Bridging the gap between raw object detection and model interpretability.

Most AI vision systems are "Black Boxes", you feed in an image and get a result, but you never see the why or how sure behind the decision. I built GlassBox AI to make the YOLOv8 detection process transparent. It doesn't just show boxes; it provides a statistical diagnostic report of the model's certainty.


🛠️ 𝗧𝗵𝗲 "𝗪𝗵𝘆" 𝗕𝗲𝗵𝗶𝗻𝗱 𝗧𝗵𝗶𝘀 𝗣𝗿𝗼𝗷𝗲𝗰𝘁:
This project was developed as a tool for AI developers to identify reliability gaps. By visualizing confidence scores across entire batches of data, we can statistically prove where a model is weak whether it's struggling with small objects, poor lighting, or specific classes like 'cell phone' or 'person.'


🚀 𝗧𝗵𝗲 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝗶𝗻𝗴 𝗖𝗵𝗮𝗹𝗹𝗲𝗻𝗴𝗲 (𝗪𝗶𝗻𝗘𝗿𝗿𝗼𝗿 𝟭𝟭𝟭𝟰)

One of the biggest hurdles during development was a local hardware conflict.

𝗧𝗵𝗲 𝗜𝘀𝘀𝘂𝗲: The application initially crashed with a WinError 1114 (DLL initialization failure) because the PyTorch backend couldn't communicate with the local hardware drivers.

𝗧𝗵𝗲 𝗙𝗶𝘅: I had to troubleshoot the entire dependency chain, isolate the environment, and optimize CUDA-DLL paths to ensure the GPU could safely initialize deep learning libraries. This experience was a deep dive into the importance of environment management in AI deployment.


📦 𝗣𝗿𝗼𝗷𝗲𝗰𝘁 𝗦𝘁𝗿𝘂𝗰𝘁𝘂𝗿𝗲

I followed a modular architecture to keep the code clean and maintainable:

  • 𝗮𝗽𝗽.𝗽𝘆: The main entry point using Streamlit for a reactive user interface.
  
  • 𝘂𝘁𝗶𝗹𝘀/𝗱𝗲𝘁𝗲𝗰𝘁𝗼𝗿.𝗽𝘆: Handles the YOLOv8 logic and hardware autodetection (CPU/GPU).
  
  • 𝘂𝘁𝗶𝗹𝘀/𝘃𝗶𝘀𝘂𝗮𝗹𝗶𝘇𝗮𝘁𝗶𝗼𝗻.𝗽𝘆: Custom Seaborn/Matplotlib logic for the Confidence Matrix.
  
  • __𝗶𝗻𝗶𝘁__.𝗽𝘆: Small but critical file that turns the utils folder into a Python package for clean imports.


🛠️ 𝗧𝗲𝗰𝗵 𝗦𝘁𝗮𝗰𝗸 & 𝗧𝗼𝗼𝗹𝘀

This project was built using a combination of modern computer vision libraries and web frameworks:

  • 𝗟𝗼𝗴𝗶𝗰 & 𝗟𝗮𝗻𝗴𝘂𝗮𝗴𝗲: Python 3.10+ (The backbone of the project).
  
  • 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻 𝗘𝗻𝗴𝗶𝗻𝗲: YOLOv8 by Ultralytics — Used for high-speed object localization and classification.
  
  • 𝗜𝗻𝘁𝗲𝗿𝗳𝗮𝗰𝗲: Streamlit: To create the reactive, web-based dashboard for real-time interaction.

𝗔𝗻𝗮𝗹𝘆𝘁𝗶𝗰𝘀:

  • 𝗦𝗲𝗮𝗯𝗼𝗿𝗻 & 𝗠𝗮𝘁𝗽𝗹𝗼𝘁𝗹𝗶𝗯: Used to generate the custom Confidence Matrix and statistical heatmaps.
  
  • 𝗡𝘂𝗺𝗣𝘆: For high-performance array manipulation of detection data.
  
  • 𝗜𝗺𝗮𝗴𝗲 𝗣𝗿𝗼𝗰𝗲𝘀𝘀𝗶𝗻𝗴: OpenCV: For handling image buffers, drawing bounding boxes, and color space conversions.

𝗕𝗮𝗰𝗸𝗲𝗻𝗱 & 𝗛𝗮𝗿𝗱𝘄𝗮𝗿𝗲:

  • 𝗣𝘆𝗧𝗼𝗿𝗰𝗵: The deep learning framework powering the model.
  
  • 𝗖𝗨𝗗𝗔: Enabled for NVIDIA GPU acceleration to achieve sub-30ms inference.
  
  • 𝗘𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁 𝗠𝗮𝗻𝗮𝗴𝗲𝗺𝗲𝗻𝘁: Anaconda / Virtualenv:  Crucial for resolving the DLL conflicts and keeping dependencies isolated.


📊 𝗛𝗼𝘄 𝗶𝘁 𝗪𝗼𝗿𝗸𝘀

  • 𝗦𝗶𝗻𝗴𝗹𝗲 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻: Quick visual check of bounding boxes and NMS accuracy.
  
  • 𝗕𝗮𝘁𝗰𝗵 𝗗𝗶𝗮𝗴𝗻𝗼𝘀𝘁𝗶𝗰: Process 20+ images at once to generate a Confidence Matrix.
  
  • 𝗧𝗵𝗿𝗲𝘀𝗵𝗼𝗹𝗱 𝗧𝘂𝗻𝗶𝗻𝗴: Use the sidebar to find the sweet spot between finding all objects (Recall) and avoiding ghosts (Precision).


🚦 𝗚𝗲𝘁𝘁𝗶𝗻𝗴 𝗦𝘁𝗮𝗿𝘁𝗲𝗱

  𝟭. 𝗖𝗹𝗼𝗻𝗲 𝘁𝗵𝗶𝘀 𝗿𝗲𝗽𝗼
  
     git clone git@github.com:mahmudaakternadia-cmd/glassbox_ai.git
     
     cd glassbox_ai
     
  𝟮. 𝗦𝗲𝘁𝘂𝗽 𝘆𝗼𝘂𝗿 𝘃𝗶𝗿𝘁𝘂𝗮𝗹 𝗲𝗻𝘃𝗶𝗿𝗼𝗻𝗺𝗲𝗻𝘁 (crucial to avoid the DLL errors I faced!).
  
      𝗕𝗮𝘀𝗵
      
      python -m venv venv
      
      source venv/Scripts/activate  # On Windows use: venv\Scripts\activate
      
  𝟯. 𝗜𝗻𝘀𝘁𝗮𝗹𝗹 𝗿𝗲𝗾𝘂𝗶𝗿𝗲𝗺𝗲𝗻𝘁𝘀: pip install -r requirements.txt.
  
  𝟰. 𝗥𝘂𝗻: streamlit run app.py.


📊 𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲 𝗘𝘃𝗮𝗹𝘂𝗮𝘁𝗶𝗼𝗻: The system was evaluated for both speed and honesty:

  • 𝗜𝗻𝗳𝗲𝗿𝗲𝗻𝗰𝗲 𝗦𝗽𝗲𝗲𝗱: Average latency of ~25ms per frame.
  
  • 𝗙𝗮𝗶𝗹𝘂𝗿𝗲 𝗠𝗮𝗽𝗽𝗶𝗻𝗴: The system intentionally flags False Negatives and low-confidence clusters, providing a clear roadmap for model fine-tuning.
