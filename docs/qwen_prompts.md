# Preparing a source image with Qwen-Image-Edit

Generic prompts for **Qwen-Image-Edit** that prepare any source image before loading it into
OversprayLab. The goal is to use the diffusion model to simplify the cutting work (cleaner layers,
fewer islands and bridges, readable key details) without losing too much of the shading, gradients
and details.

## What makes an image cut better

| Problem in the source image | Effect on the stencils | What to ask the model |
|---|---|---|
| Noise, grain, texture (skin pores, fabric, foliage) | Confetti, tiny islands, many bridges | Smooth the texture, keep the shapes |
| Irregular, blotchy gradients | Jagged layer edges | Wide, smooth, continuous gradients |
| Neighbouring areas with different hues but the same lightness | They end up in the same layer | Neighbours must differ **in lightness**, not only in hue |
| Flat lighting | Unreadable shadows, "empty" face | Strengthen a clear light direction |
| Strand-by-strand hair or fur, very fine details | Strips too thin to cut | Group into locks / masses, thicken |
| Small isolated highlights | Islands | Merge them into the neighbouring tone |
| Busy background | Useless layers, subject lost | Plain background, clearly lighter or darker than the subject's outline |
| Soft eyes, mouth, nostrils | The portrait loses its expression | Keep them crisp and marked |

The real gain of a diffusion model is that it **posterizes along the volumes and the meaning of the
image**, which a plain colour quantization cannot do.

## Common blocks (add to every prompt)

**Preservation suffix** (essential with an edit model):

```
Keep exactly the same composition, framing, pose, proportions, facial features and identity. Do not add, remove or move any element. Same aspect ratio.
```

**Negative prompt:**

```
noise, film grain, JPEG artifacts, fine texture, hatching, cross-hatching, halftone dots, dithering, speckles, tiny isolated highlights, hairline strokes, individual hair strands, cluttered background, double contours, blur, text, watermark, extra objects, changed face, distorted anatomy
```

## 0. Universal clean-up (before any style)

```
Clean up this image for stencil cutting. Remove noise, grain, compression artifacts and fine surface texture (skin pores, fabric weave, small clutter) while keeping every shape, edge and gradient. Make each gradient wide, smooth and continuous. Merge tiny specular highlights and small dark specks into the surrounding tone. Simplify the background into a smooth, low-detail area whose brightness clearly differs from the subject's outline. Keep important details crisp and well defined: eyes, eyelids, pupils, eyebrows, nostrils, lip line, fingers, main folds.
```

## 1. Realistic airbrush (4–6 tones, keeps the shading)

```
Repaint this image as a smooth, realistic airbrush illustration designed to be separated into [5] tones. Keep the volumes and soft gradients, but organize the shading into broad, readable tonal zones from highlight to deep shadow, each with a distinct brightness level. Neighbouring areas must differ in lightness, not only in hue. Strengthen the main light direction so that shadow shapes are continuous and clearly shaped. Group hair, fur and foliage into a few large locks or masses with smooth shading instead of individual strands. Slightly thicken very thin elements so they stay visible. Keep the eyes, mouth, nose and outer contours sharp.
```

## 2. Street-art stencil (2–3 tones, high contrast)

```
Turn this image into a high-contrast street-art stencil design with 3 flat tones: white, mid-grey and black. Bold, clean shapes with smooth contours, no gradients, no texture. Shadows become solid, connected dark shapes cast by a strong side light; highlights become clean light shapes. The face and key details stay recognizable through the shape of the shadows: eye sockets, pupils, nose shadow, mouth, jawline. No small isolated islands, no thin lines: every shape is wide enough to be cut. Plain background.
```

## 3. Chiaroscuro (1–2 layers, very graphic)

```
Relight this image with dramatic chiaroscuro: one strong directional light, deep connected shadows, bright lit planes. Simplify it into large, bold masses of light and dark with smooth, confident edges and very few mid-tones. The shadow side of the subject merges into a dark, plain background, while the lit side stands out clearly. Keep the eyes, nose, mouth and silhouette readable.
```

## 4. Vintage poster / screen print (stepped gradients)

```
Convert this image into a vintage screen-printed poster with a limited palette of [5] flat colours, each with a clearly different brightness. Render every gradient as a few smooth, evenly spaced bands (posterized steps) that follow the shape and volume of the forms, like contour lines on a map. Bands are wide with clean, flowing edges, never jagged or speckled. Simplify textures into flat colour areas. Keep the key facial features and the main contours precise.
```

## 5. Pop art (bold flat colours)

```
Transform this image into a bold pop-art poster: 4 flat, saturated colours chosen so that each one has a clearly different brightness (one very light, two mid, one dark), plus thick, even, continuous dark outlines around the main shapes. Shadows are simple flat shapes. No dots, no halftone, no gradients, no texture. Plain flat-colour background contrasting with the subject.
```

## 6. Geometric facets (very easy to cut)

```
Restyle this image as faceted geometric art: forms are built from large flat polygonal planes, each plane one uniform tone, arranged so that the light and shadow follow the real volumes. Use about [5] brightness levels. Facets are large near flat areas and smaller around the eyes, nose, mouth and hands so that the details stay recognizable. Straight, clean edges, no texture, simple background.
```

## 7. Comic / cel shading

```
Redraw this image in clean cel-shaded comic style: for each material, one flat base tone, one flat shadow tone and at most one small highlight tone, with crisp, smooth boundaries between them. Clean, continuous ink outlines of even weight on the main contours and key features. Hair drawn as a few large locks. No texture, no hatching, no gradients. Simple flat background.
```

## 8. Key lines (for the outline stencil)

Use the result as the source of the key-detail layer: TEED then finds continuous lines instead of
fragments.

```
Convert this image into a clean black ink line drawing on a pure white background. Keep only the meaningful lines: outer silhouette, eyes, eyelids, eyebrows, nostrils, lip line, ears, main hair locks, main clothing folds, fingers. Continuous lines of uniform, fairly bold weight, no sketchy strokes, no hatching, no shading, no texture.
```

## Usage tips

- **Chain the passes:** 0 (clean-up) then the style. A single pass tends to alter faces.
- Replace `[5]` / `[4]` with the number of layers set in OversprayLab. The model does not follow an
  exact count well, but it steers it.
- **Settings:** `true_cfg_scale` around 4, 30–50 steps. With a Lightning LoRA (4–8 steps, CFG 1)
  the negative prompt is ignored: move its important terms into the positive prompt
  ("no texture, no grain").
- **If the face changes:** add "Keep the face exactly as in the original". In ComfyUI, img2img with
  a denoise of 0.5–0.7 also keeps more of the shading.
- **Resolution:** the model outputs about 1 Mpx. That is enough for the quantization; for a large
  format, upscale afterwards (Real-ESRGAN) rather than asking the model for more.
- **Compare objectively** the raw and the prepared image in OversprayLab: number of islands and
  bridges, weak parts circled on the support map, continuity of the key-detail lines.
