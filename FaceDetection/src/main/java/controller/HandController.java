package controller;

import javafx.scene.control.Button;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import trainings.HandDetector;
import util.CameraUtil;

public class HandController {
    public Button button;
    public ImageView imageView;

    public void init() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Resource File");

        button.setOnAction(e -> {
            CameraUtil.startCamera(imageView, button);
            if(HandDetector.isDetected()){
                CameraUtil.alert();
            }
        });

    }
}
