package controller;

import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import trainings.FaceRecognition;
import trainings.HandDetector;
import util.CameraUtil;

import java.awt.image.BufferedImage;
import java.io.File;

import static java.awt.image.BufferedImage.TYPE_BYTE_GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacv.Java2DFrameUtils.toMat;

public class SceneController {
    @FXML
    public Button chooseImage;

    @FXML
    public CheckBox fishingAlgorithm;

    @FXML
    public CheckBox eigenFacesAlgorithm;

    @FXML
    public CheckBox lbphAlgorithm;

    @FXML
    public Button train;

    @FXML
    public Button predict;

    @FXML
    public Label label;

    @FXML
    public Button hand;

    @FXML
    private Button cameraButton;

    @FXML
    private ImageView originalFrame;

    @FXML
    private CheckBox haarClassifier;

    @FXML
    private CheckBox lbpClassifier;

    private final String HAAR = "src/main/resources/haarcascade_frontalface_alt.xml";

    private final String LBP = "src/main/resources/lbpcascade_frontalface.xml";

    public void init(Stage primaryStage) {
        final opencv_face.FaceRecognizer[] faceRecognizer = {null};

        label.setVisible(false);

        cameraButton.setOnAction(e -> {
            if (!haarClassifier.isSelected() && !lbpClassifier.isSelected()) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("CheckBox error");
                alert.setHeaderText("Not selected classifier");
                alert.setContentText("Choose classifier!");
                alert.showAndWait();
            } else {
                if (haarClassifier.isSelected()) {
                    CameraUtil.startCamera(originalFrame, haarClassifier, lbpClassifier, cameraButton, HAAR, false, null);
                } else {
                    CameraUtil.startCamera(originalFrame, haarClassifier, lbpClassifier, cameraButton, LBP, false, null);
                }
            }
        });

        haarClassifier.setOnAction(e -> lbpClassifier.setSelected(false));

        lbpClassifier.setOnAction(e -> haarClassifier.setSelected(false));

        setAlgoritmCheckBox(fishingAlgorithm, eigenFacesAlgorithm, lbphAlgorithm);

        setAlgoritmCheckBox(eigenFacesAlgorithm, lbphAlgorithm, fishingAlgorithm);

        setAlgoritmCheckBox(lbphAlgorithm, fishingAlgorithm, eigenFacesAlgorithm);

        train.setOnAction(e -> {
            if (!fishingAlgorithm.isSelected() && !lbphAlgorithm.isSelected() && !eigenFacesAlgorithm.isSelected()) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("CheckBox error");
                alert.setHeaderText("Not selected train algorithm");
                alert.setContentText("Choose classifier!");
                alert.showAndWait();
            } else {
                if (fishingAlgorithm.isSelected()) {
                    faceRecognizer[0] = FaceRecognition.trainAlgorithm("fish");
                } else if (eigenFacesAlgorithm.isSelected()) {
                    faceRecognizer[0] = FaceRecognition.trainAlgorithm("eigen");
                } else if (lbphAlgorithm.isSelected()) {
                    faceRecognizer[0] = FaceRecognition.trainAlgorithm("lbp");
                }
                Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
                alert.setHeaderText("SUCCESS");
                alert.showAndWait();
            }
        });

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Resource File");

        chooseImage.setOnAction(e -> {
            if (faceRecognizer[0] == null) {
                Alert alert = new Alert(Alert.AlertType.WARNING);
                alert.setHeaderText("TRAIN THE ALGORITHM");
                alert.showAndWait();
            } else {
                File file = fileChooser.showOpenDialog(primaryStage);
                Image image = new Image(file.toURI().toString());
                originalFrame.setImage(image);

                BufferedImage bufferedImage = new BufferedImage(92, 112, TYPE_BYTE_GRAY);
                bufferedImage = SwingFXUtils.fromFXImage(originalFrame.getImage(), bufferedImage);
                opencv_core.Mat mat = toMat(bufferedImage);
                cvtColor(mat, mat, COLOR_BGR2GRAY);

                double predict = FaceRecognition.predict(mat, faceRecognizer[0]);
                label.setVisible(true);
                label.setText("Prediction = " + ((predict == 100) ? "Rumy" : "Someone"));
            }
        });

        predict.setOnAction(e -> {
            if (faceRecognizer[0] == null) {
                Alert alert = new Alert(Alert.AlertType.WARNING);
                alert.setHeaderText("TRAIN THE ALGORITHM");
                alert.showAndWait();
            } else {
                if (haarClassifier.isSelected()) {
                    CameraUtil.startCamera(originalFrame, haarClassifier, lbpClassifier, cameraButton, HAAR, true, faceRecognizer[0]);
                } else {
                    CameraUtil.startCamera(originalFrame, haarClassifier, lbpClassifier, cameraButton, LBP, true, faceRecognizer[0]);
                }
            }
        });

        hand.setOnAction(e -> {
            CameraUtil.startCamera(originalFrame, hand);
            if(HandDetector.isDetected()){
                CameraUtil.alert();
            }
        });
    }

    private void setAlgoritmCheckBox(CheckBox fishingAlgorithm, CheckBox eigenFacesAlgorithm, CheckBox lbphAlgorithm) {
        fishingAlgorithm.setOnAction(e -> {
            eigenFacesAlgorithm.setSelected(false);
            lbphAlgorithm.setSelected(false);
        });
    }
}
