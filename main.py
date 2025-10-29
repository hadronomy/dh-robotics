#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import cos, sin, tan, pi, e, radians, degrees
from typing import Dict, List, Optional, Tuple
import ast
import json
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_javascript import st_javascript

matplotlib.use("Agg")


def _sind(x: float) -> float:
    return sin(radians(x))


def _cosd(x: float) -> float:
    return cos(radians(x))


def _tand(x: float) -> float:
    return tan(radians(x))


ALLOWED_FUNCS = {
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "radians": radians,
    "degrees": degrees,
    "sind": _sind,
    "cosd": _cosd,
    "tand": _tand,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}

ALLOWED_CONSTS = {"pi": pi, "e": e}


class SafeEvalError(Exception):
    pass


def safe_eval(expr: str, names: Dict[str, float]) -> float:
    try:
        node = ast.parse(expr, mode="eval")
    except Exception as ex:
        raise SafeEvalError(f"Parse error: {ex}") from ex

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise SafeEvalError("Only numeric constants are allowed.")
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.Name):
            if n.id in names:
                return float(names[n.id])
            if n.id in ALLOWED_CONSTS:
                return float(ALLOWED_CONSTS[n.id])
            raise SafeEvalError(f"Unknown variable: {n.id}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return +val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(
            n.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.Pow):
                return left**right
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise SafeEvalError("Only simple function calls are allowed.")
            fname = n.func.id
            if fname not in ALLOWED_FUNCS:
                raise SafeEvalError(f"Function not allowed: {fname}")
            args = [_eval(a) for a in n.args]
            if n.keywords:
                raise SafeEvalError("Keyword arguments are not allowed.")
            return float(ALLOWED_FUNCS[fname](*args))
        raise SafeEvalError(
            f"Disallowed syntax: {ast.dump(n, include_attributes=False)}"
        )

    return float(_eval(node))


def eval_cell(value, vars_map: Dict[str, float]) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return 0.0
    return safe_eval(s, vars_map)


def matriz_T(d: float, theta_deg: float, a: float, alpha_deg: float) -> np.ndarray:
    th = theta_deg * pi / 180.0
    al = alpha_deg * pi / 180.0
    return np.array(
        [
            [cos(th), -sin(th) * np.cos(al), sin(th) * np.sin(al), a * cos(th)],
            [sin(th), cos(th) * np.cos(al), -np.sin(al) * cos(th), a * sin(th)],
            [0.0, np.sin(al), np.cos(al), d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


@st.cache_resource(show_spinner=False)
def get_figure(figsize: Tuple[int, int]) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax


def draw_tree_scene(
    ax: plt.Axes,
    branches: List[List[np.ndarray]],
    T_dict: Dict[int, np.ndarray],
    axis_len: float,
    show_axes: bool,
    origin_flags: List[bool],
    origin_labels: Dict[int, str],
):
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    all_points = []
    for branch in branches:
        for point in branch:
            all_points.append(point[:3])
    
    if not all_points:
        all_points = [[0, 0, 0]]
    
    all_points = np.array(all_points)
    
    max_range = np.array(
        [
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min(),
        ]
    ).max()
    
    if max_range == 0:
        max_range = 1
    
    Xb = (
        0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten()
        + 0.5 * (all_points[:, 0].max() + all_points[:, 0].min())
    )
    Yb = (
        0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten()
        + 0.5 * (all_points[:, 1].max() + all_points[:, 1].min())
    )
    Zb = (
        0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten()
        + 0.5 * (all_points[:, 2].max() + all_points[:, 2].min())
    )
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")

    for branch in branches:
        if len(branch) > 0:
            branch_arr = np.array(branch)
            ax.plot3D(
                branch_arr[:, 0],
                branch_arr[:, 1],
                branch_arr[:, 2],
                marker="o",
                markersize=4,
                linewidth=2,
            )

    ax.plot3D([0], [0], [0], marker="o", color="k", ms=10)

    o_h = np.array([0.0, 0.0, 0.0, 1.0])
    
    if show_axes:
        T_base = np.eye(4)
        p = (T_base @ o_h)[:3]
        Rx = T_base[0:3, 0]
        Ry = T_base[0:3, 1]
        Rz = T_base[0:3, 2]
        ax.quiver(
            p[0], p[1], p[2],
            axis_len * Rx[0], axis_len * Rx[1], axis_len * Rx[2],
            color="r", arrow_length_ratio=0.15, linewidth=1.5,
        )
        ax.quiver(
            p[0], p[1], p[2],
            axis_len * Ry[0], axis_len * Ry[1], axis_len * Ry[2],
            color="g", arrow_length_ratio=0.15, linewidth=1.5,
        )
        ax.quiver(
            p[0], p[1], p[2],
            axis_len * Rz[0], axis_len * Rz[1], axis_len * Rz[2],
            color="b", arrow_length_ratio=0.15, linewidth=1.5,
        )
        ax.text(p[0], p[1], p[2], "O0", fontsize=9, weight="bold")
    
    for joint_idx, T in T_dict.items():
        if show_axes and origin_flags[joint_idx]:
            p = (T @ o_h)[:3]
            Rx = T[0:3, 0]
            Ry = T[0:3, 1]
            Rz = T[0:3, 2]
            ax.quiver(
                p[0], p[1], p[2],
                axis_len * Rx[0], axis_len * Rx[1], axis_len * Rx[2],
                color="r", arrow_length_ratio=0.15, linewidth=1.2,
            )
            ax.quiver(
                p[0], p[1], p[2],
                axis_len * Ry[0], axis_len * Ry[1], axis_len * Ry[2],
                color="g", arrow_length_ratio=0.15, linewidth=1.2,
            )
            ax.quiver(
                p[0], p[1], p[2],
                axis_len * Rz[0], axis_len * Rz[1], axis_len * Rz[2],
                color="b", arrow_length_ratio=0.15, linewidth=1.2,
            )
            label = origin_labels.get(joint_idx, f"J{joint_idx}")
            ax.text(p[0], p[1], p[2], label, fontsize=8)
        elif not show_axes:
            p = (T @ o_h)[:3]
            ax.plot3D([p[0]], [p[1]], [p[2]], marker="o", color="orange", markersize=5)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        return obj


REQUIRED_COLS = ["parent", "origin", "d", "theta_deg", "a", "alpha_deg"]
PRESETS_KEY = "robot_dh_presets_v2"


def ensure_dh_df(df: Optional[pd.DataFrame], n: int) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(
            {
                "parent": ["Base"] * n,
                "origin": [True] * n,
                "d": ["0"] * n,
                "theta_deg": ["0"] * n,
                "a": ["0"] * n,
                "alpha_deg": ["0"] * n,
            }
        )
        if n >= 1:
            df.loc[0, ["d", "theta_deg", "a", "alpha_deg"]] = ["L1", "0", "0", "0"]
        if n >= 2:
            df.loc[1, "parent"] = "J0"
        if n >= 3:
            df.loc[2, "parent"] = "J1"
        if n >= 4:
            df.loc[3, "parent"] = "J2"
        if n >= 5:
            df.loc[4, "parent"] = "J3"
        return df

    for c in REQUIRED_COLS:
        if c not in df.columns:
            if c == "parent":
                df[c] = "Base"
            elif c == "origin":
                df[c] = True
            else:
                df[c] = "0"

    cur = len(df)
    if n == cur:
        return df.reset_index(drop=True)
    if n < cur:
        return df.iloc[:n].reset_index(drop=True)

    extra_rows = []
    for i in range(n - cur):
        default_parent = f"J{cur + i - 1}" if (cur + i) > 0 else "Base"
        extra_rows.append({
            "parent": default_parent,
            "origin": True,
            "d": "0",
            "theta_deg": "0",
            "a": "0",
            "alpha_deg": "0",
        })
    
    extra = pd.DataFrame(extra_rows)
    out = pd.concat([df, extra], ignore_index=True)
    return out.reset_index(drop=True)


def pack_config(
    n: int,
    dh_df: pd.DataFrame,
    vars_df: pd.DataFrame,
    show_axes: bool,
    axis_len: float,
    fig_size: Tuple[int, int],
) -> Dict:
    dh_dict = {}
    for col in REQUIRED_COLS:
        if col == "origin":
            dh_dict[col] = [bool(x) for x in dh_df[col].tolist()]
        else:
            dh_dict[col] = [str(x) for x in dh_df[col].tolist()]
    
    vars_list = []
    for _, row in vars_df.iterrows():
        vars_list.append({
            "name": str(row.get("name", "")),
            "value": float(row.get("value", 0.0))
        })
    
    config = {
        "meta": {
            "version": 2,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        },
        "n": int(n),
        "dh": dh_dict,
        "vars": vars_list,
        "display": {
            "show_axes": bool(show_axes),
            "axis_len": float(axis_len),
            "fig_size": [int(fig_size[0]), int(fig_size[1])],
        },
    }
    
    return sanitize_for_json(config)


def unpack_config(cfg: Dict) -> Dict:
    n = int(cfg.get("n", 1))
    dh = cfg.get("dh", {})
    vars_list = cfg.get("vars", [])
    disp = cfg.get("display", {})
    fig_size = tuple(disp.get("fig_size", [7, 7]))
    axis_len = float(disp.get("axis_len", 2.0))
    show_axes = bool(disp.get("show_axes", True))

    dh_cols = {k: dh.get(k, []) for k in REQUIRED_COLS}
    dh_df = pd.DataFrame(dh_cols)
    
    if "origin" in dh_df:
        dh_df["origin"] = dh_df["origin"].astype(bool)
    if "parent" in dh_df:
        dh_df["parent"] = dh_df["parent"].astype(str)
    for col in ["d", "theta_deg", "a", "alpha_deg"]:
        if col in dh_df:
            dh_df[col] = dh_df[col].astype(str)

    vars_df = pd.DataFrame(vars_list, columns=["name", "value"]).fillna("")
    if "value" in vars_df.columns:
        vars_df["value"] = pd.to_numeric(vars_df["value"], errors="coerce").fillna(0.0)

    return {
        "n": n,
        "dh_df": dh_df,
        "vars_df": vars_df,
        "show_axes": show_axes,
        "axis_len": axis_len,
        "fig_size": fig_size,
    }


def parse_parent(parent_str: str) -> int:
    parent_str = str(parent_str).strip()
    if parent_str == "Base" or parent_str == "" or parent_str == "-1":
        return -1
    if parent_str.startswith("J"):
        try:
            return int(parent_str[1:])
        except ValueError:
            return -1
    try:
        idx = int(parent_str)
        return idx
    except ValueError:
        return -1


def build_origin_labels(n: int, origin_flags: List[bool]) -> Dict[int, str]:
    origin_labels = {}
    origin_counter = 1
    
    for i in range(n):
        if origin_flags[i]:
            origin_labels[i] = f"O{origin_counter}"
            origin_counter += 1
    
    return origin_labels


def build_tree_transforms(
    n: int,
    current_df: pd.DataFrame,
    T_rel_list: List[np.ndarray],
) -> Tuple[Dict[int, np.ndarray], List[List[np.ndarray]]]:
    T_dict: Dict[int, np.ndarray] = {}
    
    parents = []
    for i in range(n):
        parent_idx = parse_parent(current_df.loc[i, "parent"])
        parents.append(parent_idx)
    
    base_transform = np.eye(4)
    queue = []
    
    for i in range(n):
        if parents[i] == -1:
            T_dict[i] = base_transform @ T_rel_list[i]
            queue.append(i)
    
    processed = set([-1])
    while queue:
        parent_idx = queue.pop(0)
        if parent_idx in processed:
            continue
        processed.add(parent_idx)
        
        for i in range(n):
            if parents[i] == parent_idx and i not in T_dict:
                parent_T = T_dict.get(parent_idx, base_transform)
                T_dict[i] = parent_T @ T_rel_list[i]
                queue.append(i)
    
    for i in range(n):
        if i not in T_dict:
            T_dict[i] = base_transform @ T_rel_list[i]
    
    branches = []
    
    def build_branch_from_joint(joint_idx: int, current_path: List[np.ndarray]):
        o_h = np.array([0.0, 0.0, 0.0, 1.0])
        p = (T_dict[joint_idx] @ o_h)[:3]
        current_path.append(p.tolist())
        
        children = [i for i in range(n) if parents[i] == joint_idx]
        
        if not children:
            branches.append(current_path.copy())
        else:
            for child_idx in children:
                build_branch_from_joint(child_idx, current_path.copy())
    
    base_point = [0.0, 0.0, 0.0]
    root_joints = [i for i in range(n) if parents[i] == -1]
    
    for root_idx in root_joints:
        build_branch_from_joint(root_idx, [base_point])
    
    if not branches:
        branches = [[[0.0, 0.0, 0.0]]]
    
    return T_dict, branches


def load_presets_from_storage() -> List[Dict]:
    try:
        js_code = f"""
        (function() {{
            try {{
                const data = window.localStorage.getItem('{PRESETS_KEY}');
                return data ? JSON.parse(data) : [];
            }} catch (e) {{
                console.error('Error loading presets:', e);
                return [];
            }}
        }})()
        """
        result = st_javascript(js_code)
        
        if result is None:
            return []
        if isinstance(result, str):
            try:
                return json.loads(result)
            except:
                return []
        if isinstance(result, list):
            return result
        return []
    except Exception as e:
        st.warning(f"Could not load presets from browser storage: {e}")
        return []


def save_presets_to_storage(presets: List[Dict]) -> bool:
    try:
        clean_presets = sanitize_for_json(presets)
        presets_json = json.dumps(clean_presets, cls=NumpyEncoder)
        
        js_code = f"""
        (function() {{
            try {{
                const data = {presets_json};
                window.localStorage.setItem('{PRESETS_KEY}', JSON.stringify(data));
                return true;
            }} catch (e) {{
                console.error('Error saving presets:', e);
                return false;
            }}
        }})()
        """
        result = st_javascript(js_code)
        return result == True
    except Exception as e:
        st.error(f"Could not save presets to browser storage: {e}")
        return False


st.set_page_config(page_title="Serial Robot — Live DH Editor", layout="wide")

if "dh_df" not in st.session_state:
    st.session_state.dh_df = None
if "var_df" not in st.session_state:
    st.session_state.var_df = pd.DataFrame(
        {"name": ["L1", "Theta"], "value": [5.0, 0.0]}
    )
if "last_n" not in st.session_state:
    st.session_state.last_n = 5
if "presets" not in st.session_state:
    st.session_state.presets = []
if "presets_loaded" not in st.session_state:
    st.session_state.presets_loaded = False
if "preset_action" not in st.session_state:
    st.session_state.preset_action = None

header = st.container()
with header:
    st.title("Serial Manipulator — Live Editor with Branching")
    st.caption(
        "Edit the DH table, visualize the robot (supports tree structures), and read the origins live. "
        "Cells accept variables and expressions (e.g., L1+10, Theta+2, 2*pi, cosd(30)). "
        "Use the 'parent' column to create branches (e.g., 'Base', 'J0', 'J1', etc.)."
    )

with st.sidebar:
    st.header("Display")
    show_axes = st.toggle("Show local frames", value=True, key="show_axes")
    axis_len = st.slider("Axis length", 0.2, 5.0, 2.0, 0.1, key="axis_len")

    size_options = {
        "6×6": (6, 6),
        "7×7": (7, 7),
        "8×8": (8, 8),
        "10×10": (10, 10),
    }
    size_label = st.selectbox(
        "Figure size", options=list(size_options.keys()), index=1, key="fig_size"
    )
    fig_size = size_options[size_label]

    st.divider()
    st.header("Kinematic chain")
    n = int(st.number_input("Number of joints", 1, 50, st.session_state.last_n, key="num_joints"))
    
    if n != st.session_state.last_n:
        # Capture current edits from the data_editor before resizing
        if "dh_editor_live" in st.session_state:
            editor_value = st.session_state["dh_editor_live"]
            if isinstance(editor_value, pd.DataFrame):
                st.session_state.dh_df = editor_value.copy()
            elif isinstance(editor_value, dict) and "edited_rows" in editor_value:
                if st.session_state.dh_df is not None:
                    for idx, changes in editor_value["edited_rows"].items():
                        for col, val in changes.items():
                            st.session_state.dh_df.loc[idx, col] = val
        
        st.session_state.dh_df = ensure_dh_df(st.session_state.dh_df, n)
        st.session_state.last_n = n

    st.divider()
    st.header("Variables")
    
    var_df_edit = st.data_editor(
        st.session_state.var_df,
        use_container_width=True,
        num_rows="dynamic",
        key="vars_editor",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn(
                "name", help="Variable name (letters, digits, underscore)"
            ),
            "value": st.column_config.NumberColumn(
                "value", help="Numeric value", step=0.1
            ),
        },
    )
    
    vars_map: Dict[str, float] = {}
    for _, row in var_df_edit.iterrows():
        name = str(row.get("name", "")).strip()
        if name:
            try:
                vars_map[name] = float(row.get("value", 0.0))
            except Exception:
                pass
    vars_map.update(ALLOWED_CONSTS)

    st.divider()
    st.header("Import / Export / Presets")

    if st.session_state.dh_df is None:
        st.session_state.dh_df = ensure_dh_df(None, n)

    export_dh_df = st.session_state.dh_df
    if "dh_editor_live" in st.session_state:
        editor_value = st.session_state["dh_editor_live"]
        if isinstance(editor_value, pd.DataFrame):
            export_dh_df = editor_value
        elif isinstance(editor_value, dict) and "edited_rows" in editor_value:
            export_dh_df = st.session_state.dh_df.copy()
            for idx, changes in editor_value["edited_rows"].items():
                for col, val in changes.items():
                    export_dh_df.loc[idx, col] = val

    current_cfg = pack_config(
        n=n,
        dh_df=export_dh_df,
        vars_df=var_df_edit,
        show_axes=show_axes,
        axis_len=axis_len,
        fig_size=fig_size,
    )

    config_json = json.dumps(current_cfg, indent=2, cls=NumpyEncoder)
    st.download_button(
        "Export current config (JSON)",
        data=config_json,
        file_name=f"robot_dh_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded = st.file_uploader(
        "Import config (JSON)", type=["json"], accept_multiple_files=False, key="file_uploader"
    )
    if uploaded is not None:
        try:
            cfg_in = json.load(uploaded)
            shaped = unpack_config(cfg_in)
            st.session_state.dh_df = ensure_dh_df(shaped["dh_df"], shaped["n"])
            st.session_state.var_df = shaped["vars_df"]
            st.session_state.last_n = shaped["n"]
            st.session_state.show_axes = shaped["show_axes"]
            st.session_state.axis_len = shaped["axis_len"]
            inv_map = {v: k for k, v in size_options.items()}
            st.session_state.fig_size = inv_map.get(
                tuple(shaped["fig_size"]), "7×7"
            )
            st.success("✅ Configuration imported successfully!", icon="✅")
            st.rerun()
        except Exception as ex:
            st.error(f"❌ Invalid JSON config: {ex}")

    st.markdown("**Presets** (saved in your browser)")
    
    if not st.session_state.presets_loaded:
        with st.spinner("Loading presets from browser..."):
            st.session_state.presets = load_presets_from_storage()
            st.session_state.presets_loaded = True
    
    preset_name = st.text_input(
        "Preset name", 
        value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        key="preset_name_input"
    )
    
    cols_p = st.columns([1, 1])
    with cols_p[0]:
        if st.button("💾 Save preset", use_container_width=True, key="save_preset_btn"):
            try:
                entry = {
                    "name": preset_name.strip() or datetime.now().isoformat(),
                    "data": current_cfg,
                }
                new_presets = [p for p in st.session_state.presets if p.get("name") != entry["name"]]
                new_presets.append(entry)
                
                if save_presets_to_storage(new_presets):
                    st.session_state.presets = new_presets
                    st.success("✅ Preset saved to browser!", icon="💾")
                else:
                    st.error("❌ Could not save preset. Check browser console.")
            except Exception as e:
                st.error(f"❌ Error saving preset: {e}")
    
    with cols_p[1]:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_presets_btn"):
            st.session_state.presets = load_presets_from_storage()
            st.session_state.presets_loaded = True
            st.success("🔄 Presets refreshed!")
            st.rerun()
    
    if st.session_state.presets:
        names = [p.get("name", f"preset-{i}") for i, p in enumerate(st.session_state.presets)]
        sel = st.selectbox("Select preset", options=names, index=0, key="preset_selector")
        
        cols_p2 = st.columns([1, 1])
        with cols_p2[0]:
            if st.button("📥 Load", use_container_width=True, key="load_preset_btn"):
                try:
                    chosen = next(p for p in st.session_state.presets if p.get("name") == sel)
                    shaped = unpack_config(chosen["data"])
                    st.session_state.dh_df = ensure_dh_df(shaped["dh_df"], shaped["n"])
                    st.session_state.var_df = shaped["vars_df"]
                    st.session_state.last_n = shaped["n"]
                    st.session_state.show_axes = shaped["show_axes"]
                    st.session_state.axis_len = shaped["axis_len"]
                    inv_map = {v: k for k, v in size_options.items()}
                    st.session_state.fig_size = inv_map.get(
                        tuple(shaped["fig_size"]), "7×7"
                    )
                    st.success("✅ Preset loaded successfully!", icon="📥")
                    st.rerun()
                except Exception as ex:
                    st.error(f"❌ Could not load preset: {ex}")
        
        with cols_p2[1]:
            if st.button("🗑️ Delete", use_container_width=True, key="delete_preset_btn"):
                try:
                    new_presets = [p for p in st.session_state.presets if p.get("name") != sel]
                    if save_presets_to_storage(new_presets):
                        st.session_state.presets = new_presets
                        st.success("🗑️ Preset deleted!", icon="🗑️")
                        st.rerun()
                    else:
                        st.error("❌ Could not delete preset.")
                except Exception as e:
                    st.error(f"❌ Error deleting preset: {e}")
    else:
        st.info("No presets saved yet. Save your first preset above!", icon="ℹ")

if st.session_state.dh_df is None:
    st.session_state.dh_df = ensure_dh_df(None, n)

st.info(
    "**Parent column**: Specify where each joint connects to. Use 'Base' for base connection, "
    "'J0' for joint 0, 'J1' for joint 1, etc. This allows creating tree structures with branches. "
    "**Origin checkbox**: Only joints with 'origin' checked will be labeled as O1, O2, O3, etc. in the visualization.",
    icon="ℹ"
)

bulk = st.container()
with bulk:
    with st.form("bulk_actions_form", clear_on_submit=True):
        st.caption("Bulk origin actions:")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            select_all = st.form_submit_button("✓ Select all", use_container_width=True)
        with c2:
            clear_all = st.form_submit_button("✗ Clear all", use_container_width=True)
        with c3:
            invert = st.form_submit_button("⇄ Invert", use_container_width=True)
        with c4:
            st.form_submit_button("Cancel", use_container_width=True)
        
        if select_all:
            st.session_state.dh_df["origin"] = True
            st.rerun()
        elif clear_all:
            st.session_state.dh_df["origin"] = False
            st.rerun()
        elif invert:
            st.session_state.dh_df["origin"] = ~st.session_state.dh_df["origin"].astype(bool)
            st.rerun()

parent_options = ["Base"] + [f"J{i}" for i in range(n)]

current_df = st.data_editor(
    st.session_state.dh_df,
    num_rows="fixed",
    use_container_width=True,
    key="dh_editor_live",
    hide_index=True,
    column_order=["parent", "origin", "d", "theta_deg", "a", "alpha_deg"],
    column_config={
        "parent": st.column_config.SelectboxColumn(
            "parent",
            help="Parent joint: 'Base' for base, 'J0' for joint 0, etc.",
            options=parent_options,
            required=True,
        ),
        "origin": st.column_config.CheckboxColumn(
            "origin",
            help="Show coordinate frame for this joint (counts towards O_n numbering)",
            default=True,
        ),
        "d": st.column_config.TextColumn("d", help="Prismatic offset d"),
        "theta_deg": st.column_config.TextColumn(
            "theta_deg", help="Revolute angle θ (deg)"
        ),
        "a": st.column_config.TextColumn("a", help="Link length a"),
        "alpha_deg": st.column_config.TextColumn(
            "alpha_deg", help="Link twist α (deg)"
        ),
    },
)

current_df["parent"] = current_df["parent"].astype(str)
for col in ["d", "theta_deg", "a", "alpha_deg"]:
    current_df[col] = current_df[col].astype(str)
current_df["origin"] = current_df["origin"].astype(bool)

T_rel_list: List[np.ndarray] = []
row_errors: List[str] = []
for i in range(n):
    try:
        d_val = eval_cell(current_df.loc[i, "d"], vars_map)
        th_val = eval_cell(current_df.loc[i, "theta_deg"], vars_map)
        a_val = eval_cell(current_df.loc[i, "a"], vars_map)
        al_val = eval_cell(current_df.loc[i, "alpha_deg"], vars_map)
        T_rel_list.append(matriz_T(d_val, th_val, a_val, al_val))
    except Exception as ex:
        row_errors.append(f"Row {i+1}: {ex}")
        T_rel_list.append(np.eye(4))

T_dict, branches = build_tree_transforms(n, current_df, T_rel_list)

origin_flags = current_df["origin"].astype(bool).tolist()
origin_labels = build_origin_labels(n, origin_flags)

st.subheader("Visualization")
fig, ax = get_figure(fig_size)
draw_tree_scene(ax, branches, T_dict, axis_len, show_axes, origin_flags, origin_labels)
st.pyplot(fig, clear_figure=False)

st.subheader("Live origin values")

def fmt_vec(p):
    return [round(float(x), 6) for x in p[:3]]

o_h = np.array([0.0, 0.0, 0.0, 1.0])

origin_rows = [{"label": "O0", "joint": "Base", "parent": "-", "x": 0.0, "y": 0.0, "z": 0.0}]

for joint_idx, origin_label in sorted(origin_labels.items()):
    if joint_idx in T_dict:
        p = (T_dict[joint_idx] @ o_h)[:3]
        parent_str = current_df.loc[joint_idx, "parent"]
        origin_rows.append({
            "label": origin_label,
            "joint": f"J{joint_idx}",
            "parent": parent_str,
            "x": fmt_vec(p)[0],
            "y": fmt_vec(p)[1],
            "z": fmt_vec(p)[2],
        })

df_origins = pd.DataFrame(origin_rows, columns=["label", "joint", "parent", "x", "y", "z"])
st.dataframe(df_origins, use_container_width=True, hide_index=True)

with st.expander("Show all joint positions (J0, J1, J2, ...)"):
    joint_rows = []
    for i in range(n):
        if i in T_dict:
            p = (T_dict[i] @ o_h)[:3]
            parent_str = current_df.loc[i, "parent"]
            has_origin = "✓" if origin_flags[i] else ""
            origin_name = origin_labels.get(i, "-")
            joint_rows.append({
                "joint": f"J{i}",
                "origin": has_origin,
                "label": origin_name,
                "parent": parent_str,
                "x": fmt_vec(p)[0],
                "y": fmt_vec(p)[1],
                "z": fmt_vec(p)[2],
            })
    df_all_joints = pd.DataFrame(joint_rows, columns=["joint", "origin", "label", "parent", "x", "y", "z"])
    st.dataframe(df_all_joints, use_container_width=True, hide_index=True)

st.subheader("End effectors (leaf joints)")
parents_list = [parse_parent(current_df.loc[i, "parent"]) for i in range(n)]
leaf_joints = [i for i in range(n) if i not in parents_list]

if leaf_joints:
    ef_rows = []
    for leaf_idx in leaf_joints:
        if leaf_idx in T_dict:
            p = (T_dict[leaf_idx] @ o_h)[:3]
            origin_name = origin_labels.get(leaf_idx, "-")
            ef_rows.append({
                "joint": f"J{leaf_idx}",
                "origin_label": origin_name if origin_flags[leaf_idx] else "-",
                "x": fmt_vec(p)[0],
                "y": fmt_vec(p)[1],
                "z": fmt_vec(p)[2],
            })
    df_ef = pd.DataFrame(ef_rows, columns=["joint", "origin_label", "x", "y", "z"])
    st.dataframe(df_ef, use_container_width=True, hide_index=True)
else:
    st.info("No leaf joints found.")

if row_errors:
    st.warning(
        "⚠️ Rows with errors are treated as identity for visualization:\n"
        + "\n".join(f"- {msg}" for msg in row_errors),
        icon="⚠️"
    )