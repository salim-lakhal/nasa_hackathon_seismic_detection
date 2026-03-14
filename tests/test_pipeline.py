import io
import os
import sys
import types
import importlib
import pytest

# ---------------------------------------------------------------------------
# Torch availability
# ---------------------------------------------------------------------------

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


# ---------------------------------------------------------------------------
# Helpers for importing the Flask app without a real model on disk
# ---------------------------------------------------------------------------

def _import_app():
    """Import the Flask app, stubbing heavy deps if torch is absent."""
    app_path = os.path.join(
        os.path.dirname(__file__), "..", "my_model_demo", "app.py"
    )
    app_path = os.path.abspath(app_path)

    if not TORCH_AVAILABLE:
        # Build minimal torch stubs so the module-level imports in app.py succeed
        torch_stub = types.ModuleType("torch")
        torch_stub.device = lambda *a, **kw: "cpu"
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_stub.no_grad = lambda: _NoOpCtx()
        torch_stub.sigmoid = lambda x: x
        nn_stub = types.ModuleType("torch.nn")
        nn_stub.Module = object
        nn_stub.Conv2d = object
        nn_stub.Dropout = object
        nn_stub.Linear = object
        nn_stub.Sequential = object
        torch_stub.nn = nn_stub
        sys.modules.setdefault("torch", torch_stub)
        sys.modules.setdefault("torch.nn", nn_stub)

        tv_stub = types.ModuleType("torchvision")
        tv_transforms = types.ModuleType("torchvision.transforms")

        class _FakeCompose:
            def __init__(self, transforms):
                self.transforms = transforms
            def __call__(self, x):
                return x

        tv_transforms.Compose = _FakeCompose
        tv_transforms.Resize = lambda *a, **kw: (lambda x: x)
        tv_transforms.Grayscale = lambda *a, **kw: (lambda x: x)
        tv_transforms.ToTensor = lambda: (lambda x: x)
        tv_transforms.Normalize = lambda *a, **kw: (lambda x: x)

        tv_models = types.ModuleType("torchvision.models")
        tv_models.resnet18 = lambda **kw: None
        tv_stub.transforms = tv_transforms
        tv_stub.models = tv_models
        sys.modules.setdefault("torchvision", tv_stub)
        sys.modules.setdefault("torchvision.transforms", tv_transforms)
        sys.modules.setdefault("torchvision.models", tv_models)

    spec = importlib.util.spec_from_file_location("app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NoOpCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

@requires_torch
class TestSeismicCNN:
    def setup_method(self):
        from src.models.cnn import SeismicCNN
        self.model = SeismicCNN(num_classes=1)

    def test_forward_output_shape(self):
        x = torch.zeros(2, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (2, 1)

    def test_custom_num_classes(self):
        from src.models.cnn import SeismicCNN
        model = SeismicCNN(num_classes=3)
        x = torch.zeros(1, 1, 224, 224)
        assert model(x).shape == (1, 3)

    def test_has_trainable_params(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 0

    def test_param_count_reasonable(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert 50_000 < total < 5_000_000


@requires_torch
class TestResNet18Seismic:
    def setup_method(self):
        from src.models.cnn import ResNet18Seismic
        self.model = ResNet18Seismic(num_classes=1, pretrained=False)

    def test_forward_output_shape(self):
        x = torch.zeros(2, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (2, 1)

    def test_accepts_grayscale_input(self):
        x = torch.zeros(1, 1, 112, 112)
        out = self.model(x)
        assert out.shape == (1, 1)

    def test_param_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        assert total > 1_000_000

    def test_freeze_backbone_leaves_fc_trainable(self):
        from src.models.cnn import ResNet18Seismic
        model = ResNet18Seismic(num_classes=1, pretrained=False, freeze_backbone=True)
        frozen = [p for p in model.resnet.parameters() if not p.requires_grad]
        fc_trainable = all(p.requires_grad for p in model.resnet.fc.parameters())
        assert len(frozen) > 0
        assert fc_trainable

    def test_unfreeze_backbone(self):
        from src.models.cnn import ResNet18Seismic
        model = ResNet18Seismic(num_classes=1, pretrained=False, freeze_backbone=True)
        model.unfreeze_backbone()
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable


@requires_torch
class TestEfficientSeismicCNN:
    def setup_method(self):
        from src.models.cnn import EfficientSeismicCNN
        self.model = EfficientSeismicCNN(num_classes=1)

    def test_forward_output_shape(self):
        x = torch.zeros(4, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (4, 1)

    def test_param_count_lower_than_resnet(self):
        from src.models.cnn import ResNet18Seismic
        efficient_params = sum(p.numel() for p in self.model.parameters())
        resnet_params = sum(
            p.numel() for p in ResNet18Seismic(pretrained=False).parameters()
        )
        assert efficient_params < resnet_params

    def test_batch_size_one(self):
        x = torch.zeros(1, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (1, 1)


@requires_torch
def test_create_model_factory():
    from src.models.cnn import create_model

    for model_type in ("custom_cnn", "efficient_cnn"):
        m = create_model(model_type, num_classes=1)
        x = torch.zeros(1, 1, 224, 224)
        assert m(x).shape == (1, 1)

    with pytest.raises(ValueError):
        create_model("unknown_arch")


# ---------------------------------------------------------------------------
# Flask app tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flask_client():
    app_module = _import_app()
    flask_app = app_module.app

    # Point the Jinja loader at the real templates directory
    demo_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "my_model_demo")
    )
    flask_app.template_folder = os.path.join(demo_dir, "templates")
    flask_app.config["TESTING"] = True
    flask_app.config["UPLOAD_FOLDER"] = "/tmp/seismic_test_uploads"
    os.makedirs("/tmp/seismic_test_uploads", exist_ok=True)
    app_module.model = None
    with flask_app.test_client() as client:
        yield client


class TestFlaskRoutes:
    def test_home_returns_200(self, flask_client):
        resp = flask_client.get("/")
        assert resp.status_code == 200

    def test_upload_no_file_returns_error(self, flask_client):
        resp = flask_client.post("/upload", data={})
        assert resp.status_code == 200
        assert b"No file" in resp.data

    def test_upload_empty_filename_returns_error(self, flask_client):
        data = {"file": (io.BytesIO(b""), "")}
        resp = flask_client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        assert b"No file selected" in resp.data

    def test_upload_invalid_extension_returns_error(self, flask_client):
        data = {"file": (io.BytesIO(b"data"), "test.csv")}
        resp = flask_client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        assert b"Invalid file type" in resp.data


# ---------------------------------------------------------------------------
# allowed_file tests
# ---------------------------------------------------------------------------

class TestAllowedFile:
    @pytest.fixture(autouse=True)
    def load_module(self):
        self.app_module = _import_app()

    def test_png_allowed(self):
        assert self.app_module.allowed_file("spectrogram.png") is True

    def test_jpg_allowed(self):
        assert self.app_module.allowed_file("spectrogram.jpg") is True

    def test_jpeg_allowed(self):
        assert self.app_module.allowed_file("spectrogram.jpeg") is True

    def test_csv_not_allowed(self):
        assert self.app_module.allowed_file("data.csv") is False

    def test_exe_not_allowed(self):
        assert self.app_module.allowed_file("malware.exe") is False

    def test_no_extension_not_allowed(self):
        assert self.app_module.allowed_file("noextension") is False

    def test_case_insensitive(self):
        assert self.app_module.allowed_file("IMAGE.PNG") is True
        assert self.app_module.allowed_file("IMAGE.JPG") is True


# ---------------------------------------------------------------------------
# Fallback detection tests
# ---------------------------------------------------------------------------

class TestFallbackDetection:
    @pytest.fixture(autouse=True)
    def load_module(self):
        self.app_module = _import_app()

    def _make_grayscale_png(self, path, intensity=200):
        import numpy as np
        import cv2
        img = np.full((64, 64), intensity, dtype=np.uint8)
        cv2.imwrite(path, img)

    def test_fallback_returns_tuple(self, tmp_path):
        img_path = str(tmp_path / "flat.png")
        self._make_grayscale_png(img_path, intensity=200)
        label, confidence = self.app_module.fallback_detection(img_path)
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0

    def test_fallback_missing_image(self, tmp_path):
        label, confidence = self.app_module.fallback_detection(
            str(tmp_path / "nonexistent.png")
        )
        assert "Error" in label
        assert confidence == 0.0

    def test_fallback_high_variance_detected(self, tmp_path):
        import numpy as np
        import cv2
        img_path = str(tmp_path / "noisy.png")
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        label, confidence = self.app_module.fallback_detection(img_path)
        assert "Seismic Event Detected" in label

    def test_fallback_low_variance_no_event(self, tmp_path):
        img_path = str(tmp_path / "flat.png")
        self._make_grayscale_png(img_path, intensity=200)
        label, _ = self.app_module.fallback_detection(img_path)
        assert "No Seismic Event Detected" in label


# ---------------------------------------------------------------------------
# Transform pipeline tests
# ---------------------------------------------------------------------------

@requires_torch
class TestTransformPipeline:
    def test_transform_output_shape(self):
        from torchvision import transforms
        from PIL import Image
        import numpy as np

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        img = Image.fromarray(
            (np.random.rand(128, 128, 3) * 255).astype("uint8")
        )
        tensor = transform(img)
        assert tensor.shape == (1, 224, 224)

    def test_transform_value_range(self):
        from torchvision import transforms
        from PIL import Image
        import numpy as np

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        img = Image.fromarray(
            (np.zeros((64, 64, 3))).astype("uint8")
        )
        tensor = transform(img)
        assert float(tensor.min()) >= -1.1
        assert float(tensor.max()) <= 1.1
