# import streamlit as st
# import requests
# import tempfile
# import os

# st.set_page_config(page_title="🏡 House Construction Cost Predictor")
# st.title("🏡 House Construction Cost Predictor")

# st.markdown("""
# ### Upload your floorplan image
# - Click the **large + button** or **drag and drop** your file here.
# - Supported formats: PNG, JPG, JPEG
# """)

# uploaded_file = st.file_uploader(
#     "Upload Floorplan Image",
#     type=["png", "jpg", "jpeg"],
#     accept_multiple_files=False,
#     key="fileUploader",
#     label_visibility="collapsed"
# )

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         temp_image_path = tmp_file.name

#     st.subheader("2D to 3D Model Conversion")
#     st.markdown("""
#     When you click "Convert to 3D Model", your image will be sent to the backend, the 3D model will be generated automatically, and the result will be shown below.
#     """)

#     if st.button("Convert to 3D Model"):
#         with st.spinner("Generating 3D model, please wait..."):
#             try:
#                 with open(temp_image_path, "rb") as f:
#                     files = {"file": (os.path.basename(temp_image_path), f, "image/png")}
#                     response = requests.post("http://127.0.0.1:5000/generate-3d", files=files, timeout=180)
#                 if response.status_code == 200:
#                     data = response.json()
#                     ply_path = data.get("3d_model")
#                     if ply_path:
#                         download_url = f"http://127.0.0.1:5000{ply_path}"
#                         st.download_button(
#                             label="Download 3D Model (.ply)",
#                             data=requests.get(download_url).content,
#                             file_name="floorplan_3d_model.ply",
#                             mime="application/octet-stream"
#                         )
#                         st.markdown("#### 3D Model Viewer (Professional In-App Experience)")
#                         viewer_html = f"""
#                         <div id="viewer" style="width:100%;height:600px;"></div>
#                         <script src="https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.min.js"></script>
#                         <script src="https://cdn.jsdelivr.net/npm/three@0.150.1/examples/js/loaders/PLYLoader.js"></script>
#                         <script src="https://cdn.jsdelivr.net/npm/three@0.150.1/examples/js/controls/OrbitControls.js"></script>
#                         <script>
#                         var scene = new THREE.Scene();
#                         var camera = new THREE.PerspectiveCamera(75, 1.6, 0.1, 1000);
#                         var renderer = new THREE.WebGLRenderer({{antialias:true}});
#                         renderer.setClearColor(0xf0f0f0);
#                         renderer.setSize(window.innerWidth*0.7, 600);
#                         document.getElementById('viewer').appendChild(renderer.domElement);

#                         var controls = new THREE.OrbitControls(camera, renderer.domElement);

#                         var loader = new THREE.PLYLoader();
#                         loader.load('{download_url}', function (geometry) {{
#                             geometry.computeVertexNormals();
#                             var material = new THREE.MeshStandardMaterial({{
#                                 color: 0xdddddd,
#                                 flatShading: true,
#                                 side: THREE.DoubleSide
#                             }});
#                             var mesh = new THREE.Mesh(geometry, material);
#                             scene.add(mesh);
#                             mesh.geometry.center();
#                             camera.position.set(0, 0, 2.5);
#                             controls.update();
#                             animate();
#                         }});
#                         var light = new THREE.DirectionalLight(0xffffff, 1);
#                         light.position.set(1,1,2);
#                         scene.add(light);
#                         var ambient = new THREE.AmbientLight(0x404040, 2);
#                         scene.add(ambient);

#                         function animate() {{
#                             requestAnimationFrame(animate);
#                             controls.update();
#                             renderer.render(scene, camera);
#                         }}
#                         </script>
#                         """
#                         st.components.v1.html(viewer_html, height=620)
#                         st.info("You can also download and view the .ply file in MeshLab, Blender, or other tools.")
#                     else:
#                         st.error("3D model file not found or not generated.")
#                 else:
#                     try:
#                         err = response.json()
#                         st.error(f"3D model generation failed: {err.get('error', response.text)}")
#                         if 'stderr' in err:
#                             st.error(f"STDERR: {err['stderr']}")
#                         if 'stdout' in err:
#                             st.error(f"STDOUT: {err['stdout']}")
#                         if 'traceback' in err:
#                             st.error(f"Traceback: {err['traceback']}")
#                     except Exception:
#                         st.error(f"3D model generation failed: {response.text}")
#             except requests.exceptions.Timeout:
#                 st.error("3D model generation timed out. Try a smaller image.")
#             except requests.exceptions.ConnectionError:
#                 st.error("Could not connect to backend. Make sure the Flask server is running.")
#             except Exception as e:
#                 st.error(f"Unexpected error: {e}")
#     try:
#         os.remove(temp_image_path)
#     except Exception:
#         pass

# st.subheader("Estimate Construction Cost")
# area = st.number_input("Enter Area (m²)", min_value=10)
# rooms = st.number_input("Number of Rooms", min_value=1)
# bathrooms = st.number_input("Number of Bathrooms", min_value=1)
# if st.button("Estimate Cost"):
#     input_data = {"Area_m2": area, "Rooms": rooms, "Bathrooms": bathrooms}
#     response = requests.post("http://127.0.0.1:5000/estimate-cost", json=input_data)
#     if response.status_code == 200:
#         cost = response.json().get('estimated_cost', "Error")
#         st.success(f"🏗️ Estimated Construction Cost: ${cost}")
#     else:
#         st.error("Cost estimation failed.")

