CC = g++
PROJECT = MobileNetV1LX64
SRC = MobileNetV1LX64.cpp
OPENCV_LIBS = `pkg-config --cflags --libs opencv4`
TFLITE_LIBS = -ltensorflowlite
$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS) $(TFLITE_LIBS)