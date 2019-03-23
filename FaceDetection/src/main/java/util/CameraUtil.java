package util;

import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import trainings.FaceDetection;
import trainings.HandDetector;

import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.javacpp.opencv_imgcodecs.imencode;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RGBA2BGR;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;

public class CameraUtil {
    private static ScheduledExecutorService timer;

    private static VideoCapture capture;

    private static boolean cameraActive;


    public static void startCamera(final ImageView originalFrame, CheckBox haarClassifier,

                                   CheckBox lbpClassifier, Button cameraButton, String s, boolean flag, opencv_face.FaceRecognizer face) {


        // set a fixed width for the frame
        capture = new VideoCapture();
        originalFrame.setFitWidth(600);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);

        if (!cameraActive) {
            // disable setting checkboxes
            haarClassifier.setDisable(true);
            lbpClassifier.setDisable(true);

            // start the video capture
            capture.open(0);

            // is the video stream available?
            if (capture.isOpened()) {
                cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = () -> {
                    Image imageToShow = grabFrame(s, flag, face);
                    originalFrame.setImage(imageToShow);

                };

                timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);


                // update the button content
                cameraButton.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            cameraActive = false;
            // update again the button content
            cameraButton.setText("Start Camera");
            // enable classifiers checkboxes
            haarClassifier.setDisable(false);
            lbpClassifier.setDisable(false);
            // enable 'New user' checkbox

            stopTimer(originalFrame);

        }
    }

    public static void startCamera(final ImageView originalFrame, Button cameraButton) {


        // set a fixed width for the frame
        capture = new VideoCapture();
        originalFrame.setFitWidth(600);
        // preserve image ratio
        originalFrame.setPreserveRatio(true);

        if (!cameraActive) {

            // start the video capture
            capture.open(0);

            // is the video stream available?
            if (capture.isOpened()) {
                cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = () -> {
                    Image imageToShow = grabFrame();
                    originalFrame.setImage(imageToShow);

                };

                timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);


                // update the button content
                cameraButton.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            cameraActive = false;
            // update again the button content
            cameraButton.setText("Hand recognition");

            stopTimer(originalFrame);

        }
    }

    private static void stopTimer(ImageView originalFrame) {
        try {
            timer.shutdown();
            timer.awaitTermination(33, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            // log the exception
            System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
        }

        // release the camera
        capture.release();
        // clean the frame
        originalFrame.setImage(null);
    }

    private static Image grabFrame() {
        // init everything
        Image imageToShow = null;
        opencv_core.Mat frame = new opencv_core.Mat();


        // check if the capture is open
        if (capture.isOpened()) {
            try {
                // read the current frame
                capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    HandDetector hand=new HandDetector(frame.rows(),frame.cols());
                    hand.update(frame);
                    cvtColor(hand.getResult(), frame, CV_RGBA2BGR);
                    // convert the Mat object (OpenCV) to Image (JavaFX)
                    imageToShow = mat2Image(frame);
                }

            } catch (Exception e) {
                // log the (full) error
                System.err.println("ERROR: " + e);
            }
        }

        return imageToShow;
    }

    private static Image grabFrame(String s, boolean flag, opencv_face.FaceRecognizer face) {
        // init everything
        Image imageToShow = null;
        opencv_core.Mat frame = new opencv_core.Mat();

        // check if the capture is open
        if (capture.isOpened()) {
            try {
                // read the current frame
                capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {

                    FaceDetection.detectAndDisplay(frame, s, flag, face);

                    // convert the Mat object (OpenCV) to Image (JavaFX)
                    imageToShow = mat2Image(frame);
                }

            } catch (Exception e) {
                // log the (full) error
                System.err.println("ERROR: " + e);
            }
        }

        return imageToShow;
    }

    private static Image mat2Image(opencv_core.Mat frame) {
        // create a temporary buffer
        ByteBuffer buffer = ByteBuffer.allocate(frame.arraySize());
        // encode the frame in the buffer, according to the PNG format
        imencode(".jpg", frame, buffer);
        // build and return an Image created from the image encoded in the
        // buffer
        return new Image(new ByteArrayInputStream(buffer.array()));
    }

    public static void alert(){
        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setHeaderText("You found a hand!");
        alert.showAndWait();
    }
}
