import controller.SceneController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage)
    {
        try
        {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("scene.fxml"));
            AnchorPane root = loader.load();
            root.setStyle("-fx-background-color: #4280f5;");
            Scene scene = new Scene(root, 800, 600);
            primaryStage.setTitle("Face Detection and Tracking");
            primaryStage.setScene(scene);
            primaryStage.show();
            SceneController controller = loader.getController();
            controller.init(primaryStage);
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
