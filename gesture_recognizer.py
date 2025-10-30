from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions, \
	GestureRecognizerResult
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import cv2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

RESULT = None
GESTURE = None

import sys
import os


def resource_path(relative_path):
	"""Get absolute path to resource, works for dev and for PyInstaller bundle"""
	return (os.path.join(sys._MEIPASS, relative_path)
			if hasattr(sys, '_MEIPASS')
			else os.path.join(os.path.abspath("."), relative_path))


def print_result(result, output_image, timestamp_ms):
	global RESULT
	RESULT = result


# print("Gesture recognition result:", result)


options = GestureRecognizerOptions(
	base_options=BaseOptions(model_asset_path=resource_path('assets/gesture_recognizer.task')),
	running_mode=VisionTaskRunningMode.LIVE_STREAM,
	num_hands=2,
	min_tracking_confidence=0.4,
	min_hand_detection_confidence=0.7,
	min_hand_presence_confidence=0.6,
	result_callback=print_result)

MARGIN = 5  # pixels
FONT_SIZE = 0.75
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (0, 0, 0)
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)


def draw_landmarks_on_image(rgb_image, detection_result):
	hand_landmarks_list = detection_result.hand_landmarks
	handedness_list = detection_result.handedness
	gesture = detection_result.gestures
	annotated_image = np.copy(rgb_image)
	
	# Loop through the detected hands to visualize.
	for idx in range(len(hand_landmarks_list)):
		hand_landmarks = hand_landmarks_list[idx]
		handedness = handedness_list[idx]
		
		# Draw the hand landmarks.
		hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		hand_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
		])
		solutions.drawing_utils.draw_landmarks(
			annotated_image,
			hand_landmarks_proto,
			solutions.hands.HAND_CONNECTIONS,
			DrawingSpec(color=(0, 255, 240)),
			DrawingSpec(color=(0, 255, 240)))
		
		# Get the top left corner of the detected hand's bounding box.
		height, width, _ = annotated_image.shape
		x_coordinates = [landmark.x for landmark in hand_landmarks]
		y_coordinates = [landmark.y for landmark in hand_landmarks]
		text_x = int(min(x_coordinates) * width)
		text_y = int(min(y_coordinates) * height) - MARGIN
	
	# Draw handedness (left or right hand) on the image.
	# 	sign = gesture[idx][0] if gesture[idx] else ''
	# 	cv2.putText(annotated_image, f"{handedness[0].category_name}:{sign.category_name if sign else ''}",
	# 				(text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
	# 				FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
	
	return annotated_image


try:
	nerd_emoji = cv2.imread(str(resource_path('assets/nerd.png')))
	thumbs_down_emoji = cv2.imread(str(resource_path("assets/thumbs down.png")))
	mog_emoji = cv2.imread(str(resource_path("assets/mog.png")))
	thumbs_up_emoji = cv2.imread(str(resource_path("assets/thumbs up.png")))
	fours_emoji = cv2.imread(str(resource_path("assets/fours.png")))
	shy_emoji = cv2.imread(str(resource_path(
		"assets/shy.jpg")))
	
	if nerd_emoji is None:
		raise FileNotFoundError("nerd.png not found")
	if thumbs_down_emoji is None:
		raise FileNotFoundError("straight.png not found")
	if mog_emoji is None:
		raise FileNotFoundError("mog.png not found")
	if thumbs_up_emoji is None:
		raise FileNotFoundError("thumbs up.png not found")
	if fours_emoji is None:
		raise FileNotFoundError("fours.png not found")
	if shy_emoji is None:
		raise FileNotFoundError("shy.jpg not found")
	
	# Resize emojis
	nerd_emoji = cv2.resize(nerd_emoji, EMOJI_WINDOW_SIZE)
	mog_emoji = cv2.resize(mog_emoji, EMOJI_WINDOW_SIZE)
	thumbs_down_emoji = cv2.resize(thumbs_down_emoji, EMOJI_WINDOW_SIZE)
	thumbs_up_emoji = cv2.resize(thumbs_up_emoji, EMOJI_WINDOW_SIZE)

except Exception as e:
	print("Error loading emoji images!")
	print(f"Details: {e}")
	print("\nExpected files:")
	print("- nerd.png (nerd face)")
	print("- mog.png (straight face)")
	print("- thumbs up.png (hands up)")
	exit(1)

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("Error: Could not open webcam.")
	exit(1)

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

# print("Controls:")
# print("  Press 'q' to quit")
# print("  Raise hands above shoulders for hands up")
# print("  Smile for smiling emoji")
# print("  Straight face for neutral emoji")

# NOTE: OPENCV LIKES BGR NOT RGB, WHICH IS WHY WE NEED TO RECOLOR
with GestureRecognizer.create_from_options(options) as recognizer:
	while True:
		ret, img = cap.read()
		if not ret:
			break
		
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False
		
		frame_np = np.array(image)
		timestamp = int(round(time.time() * 1000))
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
		frame = mp_image.numpy_view()
		
		recognizer.recognize_async(mp_image, timestamp)
		
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
		gesture = None
		if RESULT:
			if RESULT.gestures and RESULT.gestures[0][0].category_name != 'None':
				gesture = RESULT.gestures[0][0].category_name
			else:
				if RESULT.hand_landmarks:
					if len(RESULT.hand_landmarks) == 1:
						index, thumb_tip, pink = RESULT.hand_landmarks[0][5], RESULT.hand_landmarks[0][4], \
							RESULT.hand_landmarks[0][17]
						if (index.x <= thumb_tip.x <= pink.x) or (index.x >= thumb_tip.x >= pink.x):
							gesture = 'Fours'
					elif len(RESULT.hand_landmarks) == 2:
						index1, index2, thumb1, thumb2 = RESULT.hand_landmarks[0][8], RESULT.hand_landmarks[1][8], \
							RESULT.hand_landmarks[0][4], RESULT.hand_landmarks[1][4]
						# print(abs(index1.x - index2.x), abs(thumb1.y - thumb2.y))
						if abs(index1.x - index2.x) <= 0.05 and abs(thumb1.y - thumb2.y) <= 0.05:
							gesture = 'Shy'
			image = draw_landmarks_on_image(image, RESULT)
		match gesture:
			case 'Pointing_Up':
				emoji_to_display = nerd_emoji
			case 'Thumb_Up':
				emoji_to_display = thumbs_up_emoji
			case 'Thumb_Down':
				emoji_to_display = thumbs_down_emoji
			case 'Fours':
				emoji_to_display = fours_emoji
			case 'Shy':
				emoji_to_display = shy_emoji
			case _:
				emoji_to_display = mog_emoji
		
		camera_frame_resized = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
		cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		
		# Display the resulting frame
		cv2.imshow('Camera Feed', camera_frame_resized)
		cv2.imshow('Emoji Output', emoji_to_display)
		# Hit q to stop transmitting frames
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
print(RESULT)
cap.release()
cv2.destroyAllWindows()
