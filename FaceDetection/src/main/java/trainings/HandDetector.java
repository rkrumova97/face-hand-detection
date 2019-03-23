package trainings;


import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;

import java.io.File;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.CASCADE_SCALE_IMAGE;


public class HandDetector {

    private static final float SMALLEST_AREA = 1200.0f; // ignore smaller contour areas

    private static final int kernelDist = 80;

    private static Mat hsvLower, hsvLower2;
    private static Mat hsvUpper, hsvUpper2;

    private int h, w;

    // defects data for the hand contour
    private ArrayList<Point> fingerTips;


    //flag indicates whether hand is detected
    private static boolean detected;

    // hand details
    private Point cogPt;           // center of gravity (COG) of contour
    private int innerRadius;

    private Mat resultImg;
    private Mat hsvImg;
    private Mat imgThreshed, imgThreshed2;
    private Mat kernel, kernel2;
    private MatVector contours;
    private Mat[] list;


    /******************Hand Detector*********************************/
    private CascadeClassifier palmCascade;
    private Mat grayImg;
    private Mat hist;
    private Mat mask;
    private int[] channels = {0};
    private int[] histSize = {32};
    private float[] ranges = {0f, 255.0f};
    private RectVector palms;


    /*************Static Gesture Recognition******************************/
    private StaticGesture staticGesture;
    private String[] staticGestureName = {"None", "Ready State", "Pressed State", "Zoom", "Bloom"};

    /*************Dynamic Gesture Recognition******************************/
    private DynamicGesture dynamicGesture;
    private String[] dynamicGestureName = {"None", "Move", "Hold", "Click", "Bloom"};


    public HandDetector(int height, int width) {
        h = height;
        w = width;
        resultImg = new Mat(height, width, CV_8UC4);
        imgThreshed = new Mat(height, width, CV_8UC1);
        imgThreshed2 = new Mat(height, width, CV_8UC1);

        kernel = new Mat(8, 8, CV_8U, new Scalar(1d));//opencv erode and dilate kernel
        kernel2 = new Mat(kernelDist, kernelDist, CV_8U, new Scalar(1d));

        hsvImg = new Mat(height, width, CV_8UC3);
        grayImg = new Mat(height, width, CV_8UC1);

        setHSV();

        contours = new MatVector();
        list = new Mat[2];
        cogPt = new Point();
        fingerTips = new ArrayList<Point>();

        File f = new File("src/main/resources/hand.xml");

        palmCascade = new CascadeClassifier(f.getAbsolutePath());
        palms = new RectVector();
        hist = new Mat();
        mask = new Mat();

        if (!palmCascade.load(f.getAbsolutePath())) {
            System.out.println("Can't load file!");
        }

        staticGesture = new StaticGesture();
        dynamicGesture = new DynamicGesture();

    }

    public static boolean isDetected() {
        return detected;
    }

    public void update(Mat im) {
        if (im.channels() == 3) {
            cvtColor(im, hsvImg, CV_BGR2HSV);
            cvtColor(im, grayImg, CV_BGR2GRAY);
            inRange(hsvImg, hsvLower, hsvUpper, imgThreshed);
            inRange(hsvImg, hsvLower2, hsvUpper2, imgThreshed2);
            add(imgThreshed, imgThreshed2, imgThreshed);
            imgThreshed.copyTo(imgThreshed2);
            cvtColor(im, resultImg, CV_BGR2RGBA);
        } else if (im.channels() == 1) {
            //process depth image
        }


        palmCascade.detectMultiScale(grayImg, palms, 1.1, 2, CASCADE_SCALE_IMAGE, new Size(100, 100), new Size(500, 500));

        Rect palm;
        for (int idx = 0; idx < palms.size(); idx++) {
            palm = palms.get(idx);
            rectangle(resultImg, palm, new Scalar(0, 255, 0, 0));
            detected = true;

        }
        calcHist(hsvImg, 1, channels, mask, hist, 1, histSize, ranges);


        erode(imgThreshed, imgThreshed, kernel);
        dilate(imgThreshed, imgThreshed, kernel);

        innerCircle(imgThreshed2);

        list[0] = findBiggestContour(imgThreshed);
        if (list[0] == null) {
            detected = false;
            return;
        }


        extractCog(list[0]);

        findFingerTips(list[0]);

        staticGesture.update(cogPt, fingerTips, innerRadius);
        dynamicGesture.update(staticGesture.getGesture());
        display();//display the result

    }

    private Mat findBiggestContour(Mat imgThreshed) {
        Mat bigContour = null;
        findContours(imgThreshed, contours, RETR_LIST, CHAIN_APPROX_NONE);

        float maxArea = SMALLEST_AREA;

        RotatedRect box;
        for (int idx = 0; idx < contours.size(); idx++) {
            box = minAreaRect(contours.get(idx));
            float area = box.size().height() * box.size().width();
            if (area > maxArea) {
                maxArea = area;
                bigContour = contours.get(idx);
            }
        }

        return bigContour;
    }

    private void findFingerTips(Mat approxContour) {
        Mat hull = new Mat();
        Mat defects = new Mat();
        convexHull(approxContour, hull, false, false);
        convexityDefects(approxContour, hull, defects);

        IntRawIndexer hullIdx = hull.createIndexer();
        IntRawIndexer contourIdx = approxContour.createIndexer();

        fingerTips.clear();

        int vertex = hullIdx.get(0, hull.rows() - 1);
        Point prev = new Point(contourIdx.get(0, vertex, 0), contourIdx.get(0, vertex, 1));
        for (int i = 0; i < hull.rows(); i++) {

            vertex = hullIdx.get(0, i);
            Point tip = new Point(contourIdx.get(0, vertex, 0), contourIdx.get(0, vertex, 1));

            if (tip.y() > cogPt.y() + 20) continue;//remove point below cogPt
            if (dist(prev, tip) < 40) continue;//remove too closed redundant points

            int tipAngle = kcurvature(vertex, approxContour, 40);
            if (tipAngle > 70) continue;//remove big angle

            fingerTips.add(tip);

            prev = tip;
        }

        convexHull(approxContour, hull, false, true);

        list[0] = approxContour;
        list[1] = hull;

    }

    private int angleBetween(Point tip, Point next, Point prev) {
        int angle = Math.abs((int) Math.round(
                Math.toDegrees(
                        Math.atan2(next.x() - tip.x(), next.y() - tip.y()) -
                                Math.atan2(prev.x() - tip.x(), prev.y() - tip.y()))));
        if (angle > 180) angle = 360 - angle;
        return angle;
    }

    private int kcurvature(int index, Mat contour, int k) {

        IntRawIndexer contourIdx = contour.createIndexer();
        int total = contour.rows();

        int prev = total + index - k;

        if (prev >= total) {
            prev = prev - total;
        }

        int next = index + k;
        if (next >= total) {
            next = next - total;
        }

        Point ptStart = new Point(contourIdx.get(0, prev, 0), contourIdx.get(0, prev, 1));
        Point ptEnd = new Point(contourIdx.get(0, next, 0), contourIdx.get(0, next, 1));
        Point ptFold = new Point(contourIdx.get(0, index, 0), contourIdx.get(0, index, 1));

        return angleBetween(ptFold, ptStart, ptEnd);
    }

    private void extractCog(Mat bigContour) {
        Moments m;
        m = moments(bigContour);
        double m00 = m.m00();
        double m10 = m.m10();
        double m01 = m.m01();

        if (m00 != 0) {
            µ(m00, m10, m01);
            circle(resultImg, cogPt, 6, new Scalar(0, 255, 0, 0), -1, 8, 0);
        }
    }

    private void µ(double m00, double m10, double m01) {
        int xCenter = (int) Math.round(m10 / m00);
        int yCenter = (int) Math.round(m01 / m00);
        cogPt.x(xCenter);
        cogPt.y(yCenter);
    }

    private void innerCircle(Mat eroded) {
        erode(eroded, eroded, kernel2);
        Moments m;
        m = moments(eroded, true);
        double m00 = m.m00();
        double m10 = m.m10();
        double m01 = m.m01();

        if (m00 != 0) {
            µ(m00, m10, m01);
        }
        int area = countNonZero(eroded);
        innerRadius = (int) Math.sqrt(area) / 4 + kernelDist - 25;
    }


    private int dist(Point u, Point v) {
        return Math.abs(u.x() - v.x()) + Math.abs(u.y() - v.y());
    }


    public Mat getResult() {
        return resultImg;
    }

    private void setHSV() {
        int huelower2;
        int huelower1;
        huelower2 = 180 + 10 - 25;
        huelower1 = 0;

        hsvLower = new Mat(h, w, CV_8UC3, new Scalar(huelower1, 105 - 55, 180 - 75, 0));
        hsvUpper = new Mat(h, w, CV_8UC3, new Scalar(10 + 25, 105 + 55, 180 + 75, 0));
        hsvLower2 = new Mat(h, w, CV_8UC3, new Scalar(huelower2, 105 - 55, 180 - 75, 0));
        hsvUpper2 = new Mat(h, w, CV_8UC3, new Scalar(255, 105 + 55, 180 + 75, 0));

    }


    private int getFingerNumber() {
        return fingerTips.size();
    }


    private void display() {

        circle(resultImg, cogPt, 6, new Scalar(0, 255, 0, 0), -1, 8, 0);
        circle(resultImg, cogPt, innerRadius, new Scalar(0, 0, 255, 0), 1, 8, 0);

        MatVector contourList = new MatVector(list);

        RNG rng = new RNG(123456); //openCV Random Number Generator  set seed 123456

        for (int i = 0; i < contourList.size(); i++) {
            Scalar color = new Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255), 0);
            drawContours(resultImg,
                    contourList,
                    i, color);
        }

        for (int i = 0; i < getFingerNumber(); i++) {
            circle(resultImg, fingerTips.get(i), 8, new Scalar(0, 0, 255, 0), -1, 8, 0);
            line(resultImg, fingerTips.get(i), cogPt, new Scalar(0, 255, 255, 0), 2, 8, 0);
        }

        if (staticGesture.getTipPostion() != null) {
            circle(resultImg, staticGesture.getTipPostion(), 8, new Scalar(255, 0, 0, 0), -1, 8, 0);
        }

        putText(resultImg, staticGestureName[staticGesture.getGesture()], new Point(0, 20), CV_FONT_HERSHEY_COMPLEX, 0.7, new Scalar(0, 255, 0, 0));

        putText(resultImg, dynamicGestureName[dynamicGesture.getGesture()], new Point(560, 20), CV_FONT_HERSHEY_COMPLEX, 0.7, new Scalar(255, 0, 0, 0));
    }
}

