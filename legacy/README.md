Earlier experiments, kept for reference. These are not maintained and their
file paths still assume they live in the project root.

- class_concept.py, Photo_test.py: first two-model approach (crop each person, check the crop for PPE)
- gemini_class.py, gemini_fix.py: single `.pt` model with background threads
- 1_model_ppe.py: YOLO11 NCNN pipeline at 416px
- test_model_performance.py: quick single-image prediction test
- old_models/: original PPE model and stock yolov8n.pt
- annotated_output.mp4: sample output from an early version
- dashboard/: local FastAPI violations dashboard (port 8080), replaced by the Supabase website in control/.
  Needs fastapi, uvicorn, python-multipart. Its saved alerts are in dashboard/data/ (not in git).
