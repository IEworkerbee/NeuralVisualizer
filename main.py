from NeuralNet import NeuralNet
import torch.nn as nn
import torch.optim as optim
import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

SCALE_MODE = True
REFERENCE_MODE = True# if reference mode is true, use min and max of all values vs local
COMBINED = True
NUM_LAYERS = 2

EPOCH_NUMBER = 400
HIDDEN_LAYER_SIZE = 20
INPUT_LAYER_SIZE = 10
OUTPUT_LAYER_SIZE = 1
COLOR_SCHEME = "plasma"

SCALE = 4 if not SCALE_MODE else min(math.floor(1900 / INPUT_LAYER_SIZE), math.floor(1080 / HIDDEN_LAYER_SIZE))
SCALE = SCALE if not COMBINED else math.floor(SCALE / ((INPUT_LAYER_SIZE + HIDDEN_LAYER_SIZE) / max(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))) 

print(SCALE)

FPS = 10



weights = {}

# 2. Define the hook function
def get_weights(name):
    def hook(model, input, output):
        # Detach from graph and move to CPU to prevent OOM errors
        if name not in weights:
            weights[name] = []
        weights[name].append(model.weight.data.clone())
    return hook

def evaluate(model):
  # Switch model to evaluation mode
  model.eval()

  # Dummy new data point
  data = [[0, 1, 1, 0, 1, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 1, 1]]
  data = [[float(x) for x in y] for y in data]
  new_data = torch.tensor(data)

  # Disable gradient calculation for faster inference
  with torch.no_grad():
      prediction = model(new_data)

  print("Model Prediction:", prediction)

def get_test_data(batch_size):
  X_train = []
  y_train = []

  """
  for i in range(batch_size):
     num_ones = random.randint(0,INPUT_LAYER_SIZE)
     bit_list = [float(1)] * num_ones + [float(0)] * (INPUT_LAYER_SIZE - num_ones)
     random.shuffle(bit_list)
     X_train.append(bit_list)
     y_train.append([float(num_ones)])

  X_train = torch.tensor(X_train)
  y_train = torch.tensor(y_train)   
  return X_train, y_train
  """
  return torch.tensor([[0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0]]), torch.tensor([[2.0]])

def make_video(file_name, layer_name, neuron1, neuron2):
  mini, maxi = 10000, -10000

  fps = FPS # Frames per second
  num_frames = EPOCH_NUMBER # Total number of frames in the video
  output_filename = f"{file_name}.mp4"

  pixel_scale = SCALE
  width, height = neuron1 * pixel_scale, neuron2 * pixel_scale

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

  if (REFERENCE_MODE):
     for i in range(num_frames):
        float_matrix = weights[layer_name][i].numpy()
      
        mini = min(mini, float_matrix.min())
        maxi = max(maxi, float_matrix.max())

  for i in range(num_frames):
      float_matrix = weights[layer_name][i].numpy()
      
      if (not REFERENCE_MODE):
        mini = float_matrix.min()
        maxi = float_matrix.max()
      
      normalized_matrix = (float_matrix - mini) / (maxi - mini)
      normalized_matrix = (normalized_matrix * 255).astype(np.uint8)
      colormap = plt.get_cmap(COLOR_SCHEME)
      colored_frame = colormap(normalized_matrix)
    
      colored_frame_bgr = colored_frame[:, :, :3] * 255
      colored_frame_bgr = colored_frame_bgr.astype(np.uint8)
      colored_frame_bgr = cv2.cvtColor(colored_frame_bgr, cv2.COLOR_RGB2BGR)
      upscaled_frame = cv2.resize(colored_frame_bgr, (width, height), interpolation=cv2.INTER_NEAREST)
      
      out.write(upscaled_frame)
  print(mini, " ", maxi)
  out.release()
  print(f"Video successfully saved as {output_filename}")

def combine_videos_side_by_side(video1_path, video2_path, output_path):

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_width = width1 + width2
    max_height = max(height1, height2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps1, (total_width, max_height))

    canvas1 = np.zeros((max_height, width1, 3), dtype=np.uint8)
    canvas2 = np.zeros((max_height, width2, 3), dtype=np.uint8)

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
           break

        canvas1.fill(0)
        canvas2.fill(0)

        canvas1[0:height1, 0:width1] = frame1
        canvas2[0:height2, 0:width2] = frame2

        combined_frame = cv2.hconcat([canvas1, canvas2])

        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved successfully as {output_path}")

def main():
  model = NeuralNet(input_size=INPUT_LAYER_SIZE, hidden_size=HIDDEN_LAYER_SIZE, output_size=OUTPUT_LAYER_SIZE)
  criterion = nn.MSELoss() 

  optimizer = optim.Adam(model.parameters(), lr=0.01) 

  hook_handle = model.fc1.register_forward_hook(get_weights('layer1'))
  hook_handle2 = model.fc2.register_forward_hook(get_weights('layer2'))
  for epoch in range(EPOCH_NUMBER):
      
      X_train, y_train = get_test_data(10)
      optimizer.zero_grad()
      outputs = model(X_train)
    
      loss = criterion(outputs, y_train)
      loss.backward()
      optimizer.step()
      
      if (epoch + 1) % 10 == 0:
          print(f"Epoch [{epoch+1}/{EPOCH_NUMBER}], Loss: {loss.item():.4f}")

  hook_handle.remove()
  hook_handle2.remove()
  make_video("layer1", "layer1", INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
  make_video("layer2", "layer2", HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
  if COMBINED:
    combine_videos_side_by_side('layer1.mp4', 'layer2.mp4', 'full_mind.mp4')


if __name__ == "__main__":
  main()