export default function GradCamViewer({ result }) {
  if (!result?.gradcam_png_base64) return null;

  return (
    <section className="card gradcam-card">
      <h2>Grad-CAM explanation</h2>
      <p>
        The heatmap highlights image regions that contributed strongly to the model prediction.
        It should be interpreted cautiously and does not prove clinical correctness.
      </p>
      <img
        className="gradcam-image"
        src={`data:image/png;base64,${result.gradcam_png_base64}`}
        alt="Grad-CAM heatmap overlay"
      />
    </section>
  );
}
