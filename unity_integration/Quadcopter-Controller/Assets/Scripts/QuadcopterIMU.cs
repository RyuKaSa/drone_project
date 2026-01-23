using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QuadcopterIMU : MonoBehaviour
{

    [SerializeField] private GameObject quadcopter;

    private Rigidbody quadcopterRb;

    private Vector3 gyroWorld;
    private Vector3 accWorld;

    private Vector3 gyroIMU;

    private Vector3 accIMU;

    private Vector3 lastVelocity;

    public double Timestamp { get; private set; }

    public System.Action<Vector3, Vector3, double> OnImuMeasured;

    // Start is called before the first frame update
    void Start()
    {
        quadcopterRb = quadcopter.GetComponent<Rigidbody>();

    }

    // Update is called once per frame
    void Update()
    {

    }


    void FixedUpdate()
    {
        Timestamp = Time.timeAsDouble;

        gyroIMU = transform.InverseTransformDirection(quadcopterRb.angularVelocity); // In local space

        accWorld = (quadcopterRb.velocity - lastVelocity) / Time.fixedDeltaTime;
        accWorld -= Physics.gravity;
        lastVelocity = quadcopterRb.velocity;

        accIMU = transform.InverseTransformDirection(accWorld);

        OnImuMeasured?.Invoke(accIMU, gyroIMU, Timestamp);


    }
}
