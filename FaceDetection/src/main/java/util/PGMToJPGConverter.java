package util;


import org.bytedeco.javacpp.opencv_core;

import javax.imageio.ImageIO;
import java.io.*;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacv.Java2DFrameUtils.toBufferedImage;

public class PGMToJPGConverter {

    public static void main(String[] args) {

        String csvFilePath = "R:\\Projects\\IdeaProjects\\trainings.FaceDetection\\src\\main\\resources\\TrainingData.txt";
        readCSVAndConvertPGMToJPG(csvFilePath);
        System.out.println("Image conversion done!");
    }

    private static void readCSVAndConvertPGMToJPG(String csvFilePath2) {
        BufferedReader br;
        try {
            br = new BufferedReader(new FileReader(csvFilePath2));

            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(";");
                File grayOutputFile = new File(tokens[0].substring(0, tokens[0].length() - 4) + ".jpg");
                opencv_core.Mat readImage = imread(tokens[0], 0);
                toBufferedImage(readImage);
                ImageIO.write(toBufferedImage(readImage), "jpg", grayOutputFile);
            }
        } catch (NumberFormatException | IOException e) {
            e.printStackTrace();
        }

    }
}
