using UnityEngine;
using RosSharp.RosBridgeClient;
using sensor_msgs = RosSharp.RosBridgeClient.MessageTypes.Sensor;
using std_msgs = RosSharp.RosBridgeClient.MessageTypes.Std;

public class RosCameraPublisher : MonoBehaviour
{
    [SerializeField] private QuadcopterFPVCamera fpvCamera;
    [SerializeField] private RosConnector rosConnector;
    [SerializeField] private string imageTopic = "/camera/image_raw";
    [SerializeField] private string frameId = "camera_link";

    private RosSocket rosSocket;
    private string publisherId;

    void Start()
    {
        if (rosConnector == null)
        {
            Debug.LogError("RosConnector is not assigned!");
            enabled = false;
            return;
        }

        if (fpvCamera == null)
        {
            Debug.LogError("QuadcopterFPVCamera is not assigned!");
            enabled = false;
            return;
        }

        rosSocket = rosConnector.RosSocket;
        publisherId = rosSocket.Advertise<sensor_msgs.Image>(imageTopic);
        fpvCamera.OnImageCaptured += PublishImage;
    }

    void PublishImage(Texture2D image, double timestamp)
    {
        if (image == null)
        {
            Debug.LogWarning("Received null image");
            return;
        }

        byte[] imageData = image.GetRawTextureData();

        sensor_msgs.Image msg = new sensor_msgs.Image
        {
            header = new std_msgs.Header
            {
                stamp = PublishImu.PublishMessage(), // Sync with ROS clock
                frame_id = frameId
            },
            height = (uint)image.height,
            width = (uint)image.width,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(image.width * 3),
            data = imageData
        };

        rosSocket.Publish(publisherId, msg);
    }

    private void OnDestroy()
    {
        if (fpvCamera != null)
            fpvCamera.OnImageCaptured -= PublishImage;
    }
}