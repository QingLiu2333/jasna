"""Encoding settings section."""

import logging
import threading

import customtkinter as ctk
from tkinter import filedialog

from jasna.accelerator import vendor_for_device
from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.icons import CompactSwitch, create_icon
from jasna.gui.locales import t
from jasna.gui.queues import MainThreadCalls
from jasna.gui.settings_sections.widgets import (
    add_setting_label,
    ValueOptionMenu,
    create_slider_value_label,
    get_tooltip,
)
from jasna.gui.theme import Colors, Fonts, Sizing
from jasna.media.codec_backend import (
    CODEC_BACKEND_LABELS,
    CODEC_BACKENDS,
    CodecBackend,
    set_codec_backend,
    set_decode_device,
)
from jasna.media.encoder_settings import encoder_cq_spec, validate_encoder_cq

logger = logging.getLogger(__name__)

# Display labels contain punctuation ("H.264 (AVC)"), so canonical values come
# from these maps, never from .lower() on the label.
CODEC_LABEL_TO_CANONICAL = {
    "HEVC (H.265)": "hevc",
    "H.264 (AVC)": "h264",
    "AV1": "av1",
}
CODEC_CANONICAL_TO_LABEL = {v: k for k, v in CODEC_LABEL_TO_CANONICAL.items()}


def _backend_labels() -> dict[str, str]:
    """Backend value -> localized display label."""
    labels: dict[str, str] = {}
    for value in CODEC_BACKENDS:
        text = t(f"codec_backend_{value}")
        # ``t`` echoes the key when a locale has no entry; fall back to the
        # English label shipped with the backend definitions.
        labels[value] = CODEC_BACKEND_LABELS[value] if text == f"codec_backend_{value}" else text
    return labels


def _decode_labels() -> dict[str, str]:
    """Decode-device labels: ``auto`` means "follow the encoder" here."""
    labels: dict[str, str] = {}
    for value in CODEC_BACKENDS:
        text = t(f"decode_device_{value}")
        fallback = CODEC_BACKEND_LABELS[value]
        labels[value] = fallback if text == f"decode_device_{value}" else text
    return labels


def _list_gpu_devices() -> list[dict]:
    """The CUDA devices, for the compute-device picker. Never raises."""
    try:
        from jasna.gpu_identity import list_cuda_devices

        return list_cuda_devices()
    except Exception:
        logger.debug("Could not enumerate CUDA devices", exc_info=True)
        return []


def describe_compute_device(index: int = 0) -> str:
    """One-line summary of a GPU: identity, token, and what it will run."""
    try:
        from jasna.arch_compat import profile_for
        from jasna.gpu_identity import describe_current_device

        identity = describe_current_device(index)
        profile = profile_for(identity.compute_capability, identity.name)
        lines = [
            identity.describe(),
            f"engine archive: {identity.token}",
            f"compute paths: {profile.summary()}",
        ]
        if not profile.fp16_worthwhile:
            lines.append(f"fp16 disabled: {profile.fp16_reason}")
        if not profile.block_quantized_gemm:
            lines.append(
                "FP4/FP8 LTX tensor-core GEMMs unavailable: quantized weights are "
                "dequantized before use"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("Could not describe the compute device", exc_info=True)
        return f"GPU info unavailable ({exc.__class__.__name__})"


class EncodingSection:
    def __init__(self, parent, widgets: dict, on_modified, preset_manager=None):
        self._widgets = widgets
        self._on_modified = on_modified
        self._preset_manager = preset_manager
        self._main_thread: MainThreadCalls | None = None

        section = CollapsibleSection(parent, t("section_encoding"), expanded=False)
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        # Which device does the codec work (decode + encode). Independent of the
        # compute GPU that runs the restoration models.
        backend_row = ctk.CTkFrame(inner, fg_color="transparent")
        backend_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        add_setting_label(backend_row, "codec_backend")
        self._backend_labels = _backend_labels()
        self._widgets["codec_backend"] = ValueOptionMenu(
            backend_row,
            options=self._backend_labels,
            command=self._on_codec_backend_changed,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY, width=200,
        )
        self._widgets["codec_backend"].pack(side="right")
        initial_backend = (
            preset_manager.get_codec_backend() if preset_manager is not None else "auto"
        )
        if initial_backend not in CODEC_BACKENDS:
            initial_backend = "auto"
        self._widgets["codec_backend"].set_value(initial_backend)
        set_codec_backend(initial_backend)

        # Which device decodes. Independent of the encoder so that a GPU with no
        # usable decoders (or a slow one) can be bypassed on purpose.
        decode_row = ctk.CTkFrame(inner, fg_color="transparent")
        decode_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        add_setting_label(decode_row, "decode_device")
        self._widgets["decode_device"] = ValueOptionMenu(
            decode_row,
            options=_decode_labels(),
            command=self._on_decode_device_changed,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY, width=200,
        )
        self._widgets["decode_device"].pack(side="right")
        initial_decode = (
            preset_manager.get_decode_device() if preset_manager is not None else "auto"
        )
        if initial_decode not in CODEC_BACKENDS:
            initial_decode = "auto"
        self._widgets["decode_device"].set_value(initial_decode)
        set_decode_device(initial_decode)

        # Which physical GPU runs the restoration. Machine-level, like the codec
        # settings: a preset made on a multi-GPU machine must not pin a device
        # index that means something else here. Only shown when there is more
        # than one CUDA device, since the index 0 default is then the only answer.
        self._gpu_devices = _list_gpu_devices()
        if len(self._gpu_devices) > 1:
            compute_row = ctk.CTkFrame(inner, fg_color="transparent")
            compute_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
            add_setting_label(compute_row, "compute_device")
            gpu_labels = {str(d["index"]): d["label"] for d in self._gpu_devices}
            self._widgets["compute_device"] = ValueOptionMenu(
                compute_row,
                options=gpu_labels,
                command=self._on_compute_device_changed,
                fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
                button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
                text_color=Colors.TEXT_PRIMARY, width=280,
            )
            self._widgets["compute_device"].pack(side="right")
            initial_compute = (
                str(preset_manager.get_compute_device()) if preset_manager is not None else "0"
            )
            if initial_compute not in gpu_labels:
                initial_compute = "0"
            self._widgets["compute_device"].set_value(initial_compute)
        else:
            self._widgets["compute_device"] = None

        # Read-only summary of the card this build will actually use: identity
        # token the engines are archived under, fp16/quantization support, and
        # what the arch fallbacks will be.
        self._gpu_info_label = ctk.CTkLabel(
            inner,
            text=t("gpu_info_pending"),
            text_color=Colors.TEXT_SECONDARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            justify="left",
            anchor="w",
            wraplength=520,
        )
        self._gpu_info_label.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        # Codec
        row1 = ctk.CTkFrame(inner, fg_color="transparent")
        row1.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        add_setting_label(row1, "codec")
        self._widgets["codec"] = ValueOptionMenu(
            row1,
            options=CODEC_CANONICAL_TO_LABEL,
            command=self._on_codec_changed,
            fg_color=Colors.BG_CARD, button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT, dropdown_fg_color=Colors.BG_CARD,
            text_color=Colors.TEXT_PRIMARY, width=120,
        )
        self._widgets["codec"].pack(side="right")
        self._widgets["codec"].set_value("hevc")
        self._active_codec = "hevc"
        self._cq_vendor = self._resolve_cq_vendor(initial_backend)
        self._cq_values = {
            codec: encoder_cq_spec(codec, self._cq_vendor).default
            for codec in CODEC_CANONICAL_TO_LABEL
        }

        # Quality/CQ
        row2 = ctk.CTkFrame(inner, fg_color="transparent")
        row2.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        add_setting_label(row2, "quality_cq", "encoder_cq")

        initial_cq = self._cq_values[self._active_codec]
        initial_spec = encoder_cq_spec(self._active_codec, self._cq_vendor)
        self._widgets["encoder_cq_val"] = create_slider_value_label(
            row2, str(initial_cq), 3, Colors.BG_PANEL
        )
        self._widgets["encoder_cq_val"].pack(side="right")
        self._widgets["encoder_cq"] = ctk.CTkSlider(
            row2,
            from_=initial_spec.minimum,
            to=initial_spec.maximum,
            number_of_steps=initial_spec.maximum - initial_spec.minimum,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160,
            command=self._on_cq_changed,
        )
        self._widgets["encoder_cq"].pack(side="right", padx=(0, 8))
        self._widgets["encoder_cq"].set(initial_cq)

        # Sharpening
        sharpen_row = ctk.CTkFrame(inner, fg_color="transparent")
        sharpen_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))

        add_setting_label(sharpen_row, "sharpen_strength")

        self._widgets["sharpen_strength_val"] = create_slider_value_label(
            sharpen_row, "0.00", 4, Colors.BG_PANEL
        )
        self._widgets["sharpen_strength_val"].pack(side="right")
        self._widgets["sharpen_strength"] = ctk.CTkSlider(
            sharpen_row, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=Colors.BG_CARD, progress_color=Colors.PRIMARY, button_color=Colors.PRIMARY,
            width=160,
            command=lambda v: self._widgets["sharpen_strength_val"].configure(text=f"{v:.2f}")
        )
        self._widgets["sharpen_strength"].pack(side="right", padx=(0, 8))
        self._widgets["sharpen_strength"].set(0.0)

        # Optional exact 60/59.94 -> 30/29.97 frame-rate retargeting.
        retarget_row = ctk.CTkFrame(inner, fg_color="transparent")
        retarget_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        retarget_label = ctk.CTkLabel(
            retarget_row,
            text=t("retarget_high_fps"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        retarget_label.pack(side="left")
        retarget_tip = ctk.CTkLabel(
            retarget_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        retarget_tip.pack(side="left", padx=4)
        Tooltip(retarget_tip, get_tooltip("retarget_high_fps"))
        self._widgets["retarget_high_fps"] = CompactSwitch(
            retarget_row,
            self._on_modified,
            Colors.BG_PANEL,
        )
        self._widgets["retarget_high_fps"].pack(side="right")

        # Fragmented MP4: output stays playable while it is written.
        fmp4_row = ctk.CTkFrame(inner, fg_color="transparent")
        fmp4_row.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        fmp4_label = ctk.CTkLabel(
            fmp4_row,
            text=t("fmp4"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        fmp4_label.pack(side="left")
        fmp4_tip = ctk.CTkLabel(
            fmp4_row,
            text="ⓘ",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        fmp4_tip.pack(side="left", padx=4)
        Tooltip(fmp4_tip, get_tooltip("fmp4"))
        self._widgets["fmp4"] = CompactSwitch(
            fmp4_row,
            self._on_modified,
            Colors.BG_PANEL,
        )
        self._widgets["fmp4"].pack(side="right")

        # Custom args
        row3 = ctk.CTkFrame(inner, fg_color="transparent")
        row3.pack(fill="x")

        args_label = ctk.CTkLabel(row3, text=t("custom_args"), text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_NORMAL))
        args_label.pack(side="left", anchor="w")
        args_tip = ctk.CTkLabel(row3, text="ⓘ", text_color=Colors.TEXT_PRIMARY, font=(Fonts.FAMILY, Fonts.SIZE_TINY), cursor="hand2")
        args_tip.pack(side="left", padx=4)
        Tooltip(args_tip, get_tooltip("encoder_custom_args"))

        args_row = ctk.CTkFrame(inner, fg_color="transparent")
        args_row.pack(fill="x", pady=(4, 0))
        self._widgets["encoder_custom_args"] = ctk.CTkEntry(
            args_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("placeholder_encoder_args")
        )
        self._widgets["encoder_custom_args"].pack(fill="x")

        # LUT (color correction)
        lut_row = ctk.CTkFrame(inner, fg_color="transparent")
        lut_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        add_setting_label(lut_row, "lut_path")

        lut_input_row = ctk.CTkFrame(inner, fg_color="transparent")
        lut_input_row.pack(fill="x", pady=(4, 0))
        self._widgets["lut_path"] = ctk.CTkEntry(
            lut_input_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("lut_path_placeholder"),
        )
        self._widgets["lut_path"].pack(side="left", fill="x", expand=True, padx=(0, 4))

        lut_browse_btn = ctk.CTkButton(
            lut_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_lut_path,
        )
        lut_browse_btn.pack(side="right")

        working_dir_row = ctk.CTkFrame(inner, fg_color="transparent")
        working_dir_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        add_setting_label(working_dir_row, "working_directory")

        working_dir_input_row = ctk.CTkFrame(inner, fg_color="transparent")
        working_dir_input_row.pack(fill="x", pady=(4, 0))
        self._widgets["working_directory"] = ctk.CTkEntry(
            working_dir_input_row, fg_color=Colors.BG_CARD, border_color=Colors.BORDER,
            text_color=Colors.TEXT_PRIMARY, placeholder_text=t("working_directory_placeholder"),
        )
        self._widgets["working_directory"].pack(side="left", fill="x", expand=True, padx=(0, 4))

        working_dir_browse_btn = ctk.CTkButton(
            working_dir_input_row, text="", image=create_icon("folder", 16, Colors.TEXT_PRIMARY), width=32, height=28,
            fg_color=Colors.BG_CARD, hover_color=Colors.BORDER_LIGHT, text_color=Colors.TEXT_PRIMARY,
            command=self._browse_working_directory,
        )
        working_dir_browse_btn.pack(side="right")

    def _resolve_cq_vendor(self, backend: str) -> str:
        """The codec backend CQ ranges/defaults should be read from.

        ``auto`` follows the compute GPU until a hardware probe says otherwise.
        """
        try:
            from jasna.media.codec_backend import (
                effective_codec_backend,
                normalize_codec_backend,
            )

            resolved = effective_codec_backend(
                normalize_codec_backend(backend),
                compute_vendor=str(vendor_for_device()),
            )
            return resolved.value
        except Exception:
            logger.debug("Could not resolve the CQ vendor for %r", backend, exc_info=True)
            return str(vendor_for_device())

    def _on_codec_backend_changed(self, backend: str):
        set_codec_backend(backend)
        if self._preset_manager is not None:
            self._preset_manager.set_codec_backend(backend)
        self._cq_vendor = self._resolve_cq_vendor(backend)
        self._on_modified()
        self._apply_cq_vendor_to_slider()

    def _on_decode_device_changed(self, device: str):
        set_decode_device(device)
        if self._preset_manager is not None:
            self._preset_manager.set_decode_device(device)
        self._on_modified()

    def _apply_cq_vendor_to_slider(self):
        """Re-range the CQ slider and reset its default for the active backend."""
        codec = self._active_codec
        spec = encoder_cq_spec(codec, self._cq_vendor)
        self._widgets["encoder_cq"].configure(
            from_=spec.minimum,
            to=spec.maximum,
            number_of_steps=spec.maximum - spec.minimum,
        )
        cq = self._cq_values.get(codec, spec.default)
        cq = max(spec.minimum, min(spec.maximum, int(cq)))
        self._cq_values[codec] = cq
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))

    def start_capability_probe(self) -> None:
        """Annotate the backend list with what this machine can actually encode.

        Runs once, off the UI thread: opening NVENC/QSV/AMF encoders is far too
        slow for the Tk main thread, and a host whose compute GPU has no video
        engines otherwise gives no clue which option will work.
        """
        # The GPU summary line is cheap enough for the UI thread (no encoder
        # probing), so fill it in immediately rather than after the probe.
        self.refresh_gpu_info()
        if self._main_thread is not None:
            return
        try:
            self._main_thread = MainThreadCalls(self._widgets["codec_backend"], 150)
        except Exception:
            logger.debug("Could not schedule the codec capability probe", exc_info=True)
            self._main_thread = None
            return
        threading.Thread(target=self._probe_capabilities, daemon=True, name="CodecProbe").start()

    def refresh_gpu_info(self) -> None:
        """Update the read-only GPU summary for the selected compute device."""
        widget = self._widgets.get("compute_device")
        try:
            index = int(widget.get_value()) if widget is not None else 0
        except (TypeError, ValueError):
            index = 0
        try:
            self._gpu_info_label.configure(text=describe_compute_device(index))
        except Exception:
            logger.debug("Could not update the GPU info line", exc_info=True)

    def _on_compute_device_changed(self, value: str) -> None:
        """Keep the whole process on one compute device: models, engines and GUI."""
        try:
            from jasna.gui.models import set_compute_device

            set_compute_device(int(value))
        except Exception:
            logger.debug("Could not store the compute device", exc_info=True)
        self.refresh_gpu_info()
        self._on_modified()

    def _probe_capabilities(self) -> None:
        from jasna.media.codec_backend import probe_backend_codec

        report: dict[str, str] = {}
        auto_choice: str | None = None
        try:
            for backend in CODEC_BACKENDS:
                if backend == CodecBackend.AUTO.value:
                    continue
                resolved = CodecBackend(backend)
                results = {}
                for codec in ("hevc", "h264", "av1"):
                    ok, _encoder, _reason = probe_backend_codec(resolved, codec, width=1280, height=720)
                    results[codec] = ok
                good = [c for c, ok in results.items() if ok]
                if good:
                    report[backend] = t("codec_backend_ok", codecs="/".join(good))
                else:
                    report[backend] = t("codec_backend_unavailable")
                # Auto prefers the compute GPU, then the iGPU, then software.
                if auto_choice is None and good:
                    auto_choice = backend
            if auto_choice is not None:
                report[CodecBackend.AUTO.value] = t(
                    "codec_backend_auto_pick", backend=self._backend_labels.get(auto_choice, auto_choice)
                )
        except Exception:
            logger.debug("Codec capability probe failed", exc_info=True)
            return
        main_thread = self._main_thread
        if main_thread is None:
            return
        main_thread.post(lambda: self._apply_probe_report(report, auto_choice))

    def _apply_probe_report(self, report: dict[str, str], auto_choice: str | None) -> None:
        labels = dict(self._backend_labels)
        for value, suffix in report.items():
            base = labels.get(value, value)
            # Strip any previous annotation so a re-probe does not stack text.
            base = base.split(" — ")[0]
            labels[value] = f"{base} — {suffix}"
        self._backend_labels = labels
        self._widgets["codec_backend"].set_options(labels)
        if auto_choice is not None and self._widgets["codec_backend"].get_value() == CodecBackend.AUTO.value:
            # Show the range/default of the encoder ``auto`` will really use.
            self._cq_vendor = self._resolve_cq_vendor(auto_choice)
            self._apply_cq_vendor_to_slider()

    def close(self) -> None:
        main_thread = self._main_thread
        if main_thread is not None:
            self._main_thread = None
            try:
                main_thread.close()
            except Exception:
                logger.debug("Could not cancel the codec probe timer", exc_info=True)

    def _on_codec_changed(self, new_codec: str):
        self._cq_values[self._active_codec] = int(
            self._widgets["encoder_cq"].get()
        )
        self._active_codec = new_codec
        spec = encoder_cq_spec(new_codec, self._cq_vendor)
        self._widgets["encoder_cq"].configure(
            from_=spec.minimum,
            to=spec.maximum,
            number_of_steps=spec.maximum - spec.minimum,
        )
        cq = self._cq_values[new_codec]
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._on_modified()

    def _on_cq_changed(self, value: float):
        cq = int(value)
        self._cq_values[self._active_codec] = cq
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._on_modified()

    def _browse_lut_path(self):
        filepath = filedialog.askopenfilename(
            title=t("dialog_select_lut"),
            filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")],
        )
        if filepath:
            self._widgets["lut_path"].delete(0, "end")
            self._widgets["lut_path"].insert(0, filepath)

    def _browse_working_directory(self):
        directory = filedialog.askdirectory(title=t("dialog_select_working_directory"))
        if directory:
            self._widgets["working_directory"].delete(0, "end")
            self._widgets["working_directory"].insert(0, directory)

    def apply(self, preset):
        self._widgets["codec"].set_value(preset.codec)
        self._active_codec = self._widgets["codec"].get_value()
        self._cq_values = {
            codec: encoder_cq_spec(codec, self._cq_vendor).default
            for codec in CODEC_CANONICAL_TO_LABEL
        }
        spec = encoder_cq_spec(self._active_codec, self._cq_vendor)
        cq = spec.default if preset.encoder_cq is None else preset.encoder_cq
        validate_encoder_cq(
            cq,
            codec=self._active_codec,
            vendor=self._cq_vendor,
        )
        if not isinstance(cq, int) or isinstance(cq, bool):
            raise ValueError(f"GUI CQ must be an integer (got {cq!r})")
        self._cq_values[self._active_codec] = cq
        self._widgets["encoder_cq"].configure(
            from_=spec.minimum,
            to=spec.maximum,
            number_of_steps=spec.maximum - spec.minimum,
        )
        self._widgets["encoder_cq"].set(cq)
        self._widgets["encoder_cq_val"].configure(text=str(cq))
        self._widgets["encoder_custom_args"].delete(0, "end")
        self._widgets["encoder_custom_args"].insert(0, preset.encoder_custom_args)
        self._widgets["sharpen_strength"].set(preset.sharpen_strength)
        self._widgets["sharpen_strength_val"].configure(
            text=f"{preset.sharpen_strength:.2f}"
        )
        if preset.retarget_high_fps:
            self._widgets["retarget_high_fps"].select()
        else:
            self._widgets["retarget_high_fps"].deselect()
        if preset.fmp4:
            self._widgets["fmp4"].select()
        else:
            self._widgets["fmp4"].deselect()

        self._widgets["lut_path"].delete(0, "end")
        self._widgets["lut_path"].insert(0, preset.lut_path or "")

        self._widgets["working_directory"].delete(0, "end")
        self._widgets["working_directory"].insert(0, preset.working_directory or "")

    def collect(self) -> dict:
        return {
            "codec": self._widgets["codec"].get_value(),
            "encoder_cq": int(self._widgets["encoder_cq"].get()),
            "encoder_custom_args": self._widgets["encoder_custom_args"].get(),
            "sharpen_strength": round(float(self._widgets["sharpen_strength"].get()), 2),
            "retarget_high_fps": self._widgets["retarget_high_fps"].get() == 1,
            "fmp4": self._widgets["fmp4"].get() == 1,
            "lut_path": self._widgets["lut_path"].get().strip(),
            "working_directory": self._widgets["working_directory"].get().strip(),
        }
