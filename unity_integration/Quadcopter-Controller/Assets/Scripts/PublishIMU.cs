using System;
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.RosBridgeClient.MessageTypes.Sensor;
using RosSharp.RosBridgeClient.MessageTypes.Std;
using RosVector3 = RosSharp.RosBridgeClient.MessageTypes.Geometry.Vector3;


public class PublishImu : MonoBehaviour
{
    [SerializeField] private RosConnector rosConnector;
    [SerializeField] private QuadcopterIMU imu;
    [SerializeField] private string topic = "/imu/data_raw";
    [SerializeField] private string frameId = "imu";

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

        if (imu == null)
        {
            Debug.LogError("QuadcopterIMU is not assigned!");
            enabled = false;
            return;
        }

        rosSocket = rosConnector.RosSocket;
        publisherId = rosSocket.Advertise<Imu>(topic);
        imu.OnImuMeasured += OnImuMeasured;
    }

    void OnImuMeasured(Vector3 acc, Vector3 gyro, double timestamp)
    {
        Imu msg = new Imu
        {
            header = new Header
            {
                stamp = PublishMessage(),
                frame_id = frameId
            },
            angular_velocity = new RosVector3
            {
                x = gyro.x,
                y = gyro.y,
                z = gyro.z
            },
            linear_acceleration = new RosVector3
            {
                x = acc.x,
                y = acc.y,
                z = acc.z
            }
        };

        rosSocket.Publish(publisherId, msg);
    }

    void OnDestroy()
    {
        if (imu != null)
            imu.OnImuMeasured -= OnImuMeasured;
    }

    public static (int sec, uint nanosec) PublishMessage()
    {
        double publishTime = Time.realtimeSinceStartupAsDouble;

        int sec = (int)Math.Floor(publishTime);
        uint nanosec = (uint)((publishTime - sec) * 1e9);

        return (sec, nanosec);
    }

}