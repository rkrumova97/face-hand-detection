package trainings;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_face.EigenFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_face.FisherFaceRecognizer;
import org.bytedeco.javacpp.opencv_face.LBPHFaceRecognizer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

public class FaceRecognition {
    public static FaceRecognizer trainAlgorithm(String algorithm) {

        ArrayList<Mat> images = new ArrayList<>();
        ArrayList<Integer> labels = new ArrayList<>();
        String csvFilePath = "src/main/resources/TrainingData.txt";
        readCSV(csvFilePath, images, labels);


        images.remove(images.size() - 1);
        labels.remove(labels.size() - 1);

        FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
        FaceRecognizer fish = FisherFaceRecognizer.create();
        FaceRecognizer lbp = LBPHFaceRecognizer.create();

        MatVector imagesVector = new MatVector(images.size());

        Mat label = new Mat(images.size(), 1, CV_32SC1);
        IntBuffer labelsBuf = label.createBuffer();

        for (int i = 0; i < images.size(); i++) {
            imagesVector.put(i, images.get(i));
            labelsBuf.put(i, labels.get(i));

        }

        switch (algorithm) {
            case "eigen":
                faceRecognizer.train(imagesVector, label);
                return faceRecognizer;
            case "fishing":
                fish.train(imagesVector, label);
                return fish;
            case "lbp":
                lbp.train(imagesVector, label);
                return lbp;
            default:
                return null;
        }

    }

    public static double predict(Mat testSample, FaceRecognizer fish) {

        IntPointer l = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        fish.predict(testSample, l, confidence);
        int predictedLabel = l.get(0);

        System.out.println("Predicted label: " + predictedLabel);

        return predictedLabel;
    }

    private static void readCSV(String csvFilePath2, ArrayList<Mat> images, ArrayList<Integer> labels) {
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader(csvFilePath2));

            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(";");
                Mat readImage = imread(tokens[0], 0);
                images.add(readImage);
                labels.add(Integer.parseInt(tokens[1]));
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }
}