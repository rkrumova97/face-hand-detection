package trainings;

import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;

import java.io.IOException;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.CASCADE_SCALE_IMAGE;


public class FaceDetection {

    public static void detectAndDisplay(Mat frame, String s, boolean flag, opencv_face.FaceRecognizer faceRecognition) throws IOException {
        RectVector faces = new RectVector();
        Mat grayFrame = new Mat();
        int absoluteFaceSize = 0;
        CascadeClassifier faceCascade = new CascadeClassifier();

        faceCascade.load(s);

        // convert the frame in gray scale
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (1% of the frame height, in our case)

        int height = grayFrame.rows();
        if (Math.round(height * 0.2f) > 0) {
            absoluteFaceSize = Math.round(height * 0.01f);
        }

        // detect faces
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, CASCADE_SCALE_IMAGE,
                new Size(absoluteFaceSize, absoluteFaceSize), new Size(height, height));

        // each rectangle in faces is a face: draw them!
        Rect[] facesArray = faces.get();
        System.out.println("Number of faces detected = " + facesArray.length);
        for (Rect rect : facesArray) {
            rectangle(frame, rect.tl(), rect.br(), new Scalar(0.0, 255.0, 0.0, 0.0), 0, 255, 0);
            if (flag) {
                Rect rectCrop = new Rect(rect.tl(), rect.br());
                Mat croppedImage = new Mat(frame, rectCrop);
                // Change to gray scale
                cvtColor(croppedImage, croppedImage, COLOR_RGB2GRAY);
                // Equalize histogram
                equalizeHist(croppedImage, croppedImage);
                // Resize the image to a default size
                Mat resizeImage = new Mat();
                Size size = new Size(92, 112);
                resize(croppedImage, resizeImage, size);
                double algorithm = FaceRecognition.predict(resizeImage, faceRecognition);
                int pos_x = Math.max(rect.tl().x() - 10, 0);
                int pos_y = Math.max(rect.tl().y() - 10, 0);
                putText(frame, "Prediction = " + ((algorithm == 100) ? "Rumy" : "Someone") , new Point(pos_x, pos_y),
                        FONT_HERSHEY_PLAIN, 1.0, new Scalar(0, 255, 0, 2.0));
            }
        }
    }

}

