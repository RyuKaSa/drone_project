using System;
using UnityEngine;

public class QuadcopterFPVCamera : MonoBehaviour
{
	[SerializeField] private Camera camera;
	[SerializeField] private int cameraFpsLimit = 30;
	[SerializeField] private int cameraWidth = 640;
	[SerializeField] private int cameraHeight = 480;
	[SerializeField] private bool render = true;

	private float fpsTimer;
	private RenderTexture renderTexture;
	private Texture2D captureTexture;

	// Callback pour VIO / ROS / OpenCV
	public Action<Texture2D, double> OnImageCaptured;

	private void OnGUI()
	{
		GUI.DrawTexture(
			new Rect(0, Screen.height - cameraHeight, cameraWidth, cameraHeight),
			renderTexture
		);
	}

	void Start()
	{
		renderTexture = new RenderTexture(
			cameraWidth,
			cameraHeight,
			0,
			RenderTextureFormat.RGB565
		);

		captureTexture = new Texture2D(
			cameraWidth,
			cameraHeight,
			TextureFormat.RGB24,
			false
		);

		camera.targetTexture = renderTexture;
		camera.enabled = false;
	}

	void Update()
	{
		if (!render) return;

		if (cameraFpsLimit > 0)
		{
			fpsTimer += Time.deltaTime;
			if (fpsTimer < 1f / cameraFpsLimit)
				return;

			fpsTimer = 0f;
		}

		CaptureFrame();
	}

	private void CaptureFrame()
	{
		camera.Render();

		RenderTexture.active = renderTexture;

		captureTexture.ReadPixels(
			new Rect(0, 0, cameraWidth, cameraHeight),
			0,
			0
		);

		captureTexture.Apply(false);

		RenderTexture.active = null;

		double timestamp = Time.timeAsDouble;

		OnImageCaptured?.Invoke(captureTexture, timestamp);
	}

	private void OnDestroy()
	{
		renderTexture.Release();
	}
}
