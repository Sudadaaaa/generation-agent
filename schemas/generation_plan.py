from typing import Literal, Optional

from pydantic import BaseModel, Field


class Subject(BaseModel):
    """The main subject of the generation."""

    description: str = Field(
        description="Description of the main subject."
    )

    action: Optional[str] = Field(
        default=None,
        description="Action performed by the subject."
    )

    position: Optional[str] = Field(
        default=None,
        description="Position of the subject in the frame."
    )


class Environment(BaseModel):
    """The scene and environment."""

    description: str = Field(
        description="Description of the environment or scene."
    )

    time: Optional[str] = Field(
        default=None,
        description="Time of day, such as morning, sunset, or night."
    )

    weather: Optional[str] = Field(
        default=None,
        description="Weather conditions, such as rain, snow, or clear."
    )


class Camera(BaseModel):
    """Camera and cinematography information."""

    shot_type: Optional[str] = Field(
        default=None,
        description=(
            "Shot type, such as close-up, medium shot, "
            "wide shot, or extreme wide shot."
        ),
    )

    angle: Optional[str] = Field(
        default=None,
        description="Camera angle, such as eye-level, low-angle, or high-angle.",
    )

    movement: Optional[str] = Field(
        default=None,
        description=(
            "Camera movement, such as dolly in, dolly out, "
            "pan, tilt, tracking, or static."
        ),
    )


class Composition(BaseModel):
    """Visual composition information."""

    description: Optional[str] = Field(
        default=None,
        description="Description of the overall composition."
    )

    aspect_ratio: str = Field(
        default="16:9",
        description="Desired output aspect ratio."
    )


class Style(BaseModel):
    """Visual style information."""

    description: Optional[str] = Field(
        default=None,
        description="Overall visual or artistic style."
    )

    lighting: Optional[str] = Field(
        default=None,
        description="Lighting characteristics."
    )

    color: Optional[str] = Field(
        default=None,
        description="Color palette or color characteristics."
    )


class TextElement(BaseModel):
    """Text that should appear inside the generated image or video."""

    content: str = Field(
        description="Exact text that should appear in the generated media."
    )

    position: Optional[str] = Field(
        default=None,
        description="Desired position of the text."
    )

    style: Optional[str] = Field(
        default=None,
        description="Visual style of the text."
    )


class GenerationConfig(BaseModel):
    """Generation-related configuration."""

    model: Optional[str] = Field(
        default=None,
        description="Suggested generation model."
    )

    duration: Optional[float] = Field(
        default=None,
        description="Video duration in seconds. Null for image generation."
    )


class GenerationPlan(BaseModel):
    """
    Structured representation of a user's generative media request.

    This is the central intermediate representation of the project.
    """

    task: Literal[
        "text_to_image",
        "image_to_image",
        "text_to_video",
        "image_to_video",
    ] = Field(
        description="Type of generative media task."
    )

    subject: Subject = Field(
        description="Main subject of the generation."
    )

    environment: Environment = Field(
        description="Scene and environment."
    )

    camera: Optional[Camera] = Field(
        default=None,
        description="Camera and cinematography information."
    )

    composition: Optional[Composition] = Field(
        default=None,
        description="Composition information."
    )

    style: Optional[Style] = Field(
        default=None,
        description="Visual style information."
    )

    text: Optional[TextElement] = Field(
        default=None,
        description="Text that should appear in the generated media."
    )

    generation: GenerationConfig = Field(
        description="Generation configuration."
    )