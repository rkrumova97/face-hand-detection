import controller.HandController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

public class Some extends Application {
    @Override
    public void start(Stage primaryStage)
    {
        try
        {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("hand.fxml"));
            AnchorPane root = loader.load();
            root.setStyle("-fx-background-color: whitesmoke;");
            Scene scene = new Scene(root);
            primaryStage.setTitle("Hand Detection and Tracking");
            primaryStage.setScene(scene);
            primaryStage.show();
            HandController controller = loader.getController();
            controller.init();
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }

}
