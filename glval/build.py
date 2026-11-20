"""Build an isolated, frozen runtime. AST edits only affect the new scaffold copy."""
import ast
import shutil
from pathlib import Path

from .common import ROOT, VENDOR, digest, file_hash, inventory, read_json, write_json
from .methods import check_method


class DriverBridge(ast.NodeTransformer):
    def __init__(self, workload, rolling_checkpoint=False):
        self.workload = workload
        self.rolling_checkpoint = rolling_checkpoint
        self.checkpoint_injections = 0
        self.registration_imports = 0
        self.result_injections = 0
        self.resnet_attach = 0

    def visit_ImportFrom(self, node):
        if self.workload == "resnet" and node.module in {"comm_hooks.default_hooks", "comm_hooks.subspace_hook"}:
            return None
        if node.module == "comm_hooks.utils" and any(x.name == "register_comm_hook_for_ddp_model" for x in node.names):
            self.registration_imports += 1
            rest = [x for x in node.names if x.name != "register_comm_hook_for_ddp_model"]
            nodes = [ast.ImportFrom(module="glval.bridge", names=[ast.alias(name="register_comm_hook_for_ddp_model")], level=0)]
            if rest:
                nodes.insert(0, ast.ImportFrom(module=node.module, names=rest, level=0))
            return nodes
        return node

    def visit_FunctionDef(self, node):
        if self.workload == "timing" and node.name == "register_timing_comm_hook":
            self.resnet_attach += 1
            node.args.args.append(ast.arg(arg="optimizer"))
            node.body = ast.parse(
                "from glval.bridge import register_comm_hook_for_ddp_model\n"
                "if not getattr(args, 'record_bucket_layout', False):\n"
                "    register_comm_hook_for_ddp_model(model, process_group, args, optimizer)\n"
                "    return None\n"
                "observer = BucketRecordingHookState(None, None)\n"
                "native_register = model.register_comm_hook\n"
                "def register_observed(inner_state, inner_hook):\n"
                "    observer.inner_state, observer.inner_hook = inner_state, inner_hook\n"
                "    native_register(observer, bucket_recording_hook)\n"
                "model.register_comm_hook = register_observed\n"
                "try:\n"
                "    register_comm_hook_for_ddp_model(model, process_group, args, optimizer)\n"
                "finally:\n"
                "    model.register_comm_hook = native_register\n"
                "return observer"
            ).body
            return node
        if self.workload == "resnet" and node.name == "attach_hook":
            self.resnet_attach += 1
            node.args.args.append(ast.arg(arg="optimizer"))
            node.body = ast.parse(
                "from glval.bridge import attach_resnet\nreturn attach_resnet(model, args, optimizer)"
            ).body
            return node
        return self.generic_visit(node)

    def visit_If(self, node):
        if self.workload == "llama" and any(isinstance(n, ast.Attribute) and n.attr == "save_every" for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if self.workload == "timing":
            if isinstance(node.func, ast.Name) and node.func.id == "register_timing_comm_hook":
                node.args.append(ast.Name(id="optimizer", ctx=ast.Load()))
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess" and node.func.attr == "check_output"
                    and node.args and isinstance(node.args[0], ast.List)
                    and any(isinstance(x, ast.Constant) and x.value == "git" for x in node.args[0].elts)):
                return ast.parse('__import__("json").loads(Path("method.json").read_text())["source_commit"]').body[0].value
            if isinstance(node.func, ast.Attribute) and node.func.attr == "mkdir":
                for keyword in node.keywords:
                    if keyword.arg == "exist_ok": keyword.value = ast.Constant(value=True)
        if (isinstance(node.func, ast.Name) and node.func.id == "register_comm_hook_for_ddp_model"
                and not any(k.arg == "optimizer" for k in node.keywords)):
            node.keywords.append(ast.keyword(arg="optimizer", value=ast.Name(id="optimizer", ctx=ast.Load())))
        if self.workload == "resnet":
            if isinstance(node.func, ast.Name) and node.func.id == "attach_hook":
                node.args.append(ast.Name(id="optimizer", ctx=ast.Load()))
            for keyword in node.keywords:
                if keyword.arg == "download":
                    keyword.value = ast.Constant(value=False)
        if (self.workload == "llama" and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "AutoTokenizer"
                and node.func.attr == "from_pretrained"):
            node.args[0] = ast.parse('os.environ["GLVAL_TOKENIZER_PATH"]').body[0].value
            node.keywords.append(ast.keyword(arg="local_files_only", value=ast.Constant(value=True)))
        if (self.workload == "glue" and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "raw_datasets"
                and node.func.attr == "map"):
            value = ast.parse('{split: os.path.join(os.environ["GLVAL_OUTPUT"], "tokenized-" + split + ".arrow") for split in raw_datasets}').body[0].value
            node.keywords.append(ast.keyword(arg="cache_file_names", value=value))
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if (self.workload == "llama" and self.rolling_checkpoint
                and any(isinstance(t, ast.Name) and t.id == "update_time" for t in node.targets)
                and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Sub)):
            self.checkpoint_injections += 1
            return [node, *ast.parse("from glval.checkpoints import save_if_due\nsave_if_due(locals())").body]
        if any(isinstance(t, ast.Name) and t.id == "all_results" for t in node.targets):
            self.result_injections += 1
            extra = ast.parse(
                "from glval.bridge import completion_evidence\n"
                "all_results.update(completion_evidence(locals()))"
            ).body
            return [node, *extra]
        return node


def instrument(runtime, workload, rolling_checkpoint=False):
    relative = {"llama": "c4/run_llama_pretraining.py", "resnet": "pytorch-cifar/train_ddp.py",
                "glue": "glue/run_glue_no_trainer_HF.py", "timing": "c4/table_v_timing.py"}[workload]
    path = runtime / relative
    tree = ast.parse(path.read_text())
    bridge = DriverBridge(workload, rolling_checkpoint)
    tree = ast.fix_missing_locations(bridge.visit(tree))
    if workload in {"resnet", "timing"}:
        if bridge.resnet_attach != 1:
            raise ValueError("CIFAR driver bridge anchor changed")
    elif bridge.registration_imports != 1 or bridge.result_injections != 1:
        raise ValueError("Training driver bridge anchors changed")
    if rolling_checkpoint and bridge.checkpoint_injections != 1:
        raise ValueError("Rolling checkpoint update anchor changed")
    path.write_text(ast.unparse(tree) + "\n")
    return relative


def build_runtime(target, method_path, recipe):
    method, source, adapter = check_method(method_path, recipe["workload"])
    shutil.copytree(VENDOR, target)
    for name in method["overlay"]:
        src, dst = source / name, target / name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    shutil.copytree(ROOT / "glval", target / "glval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(adapter, target / "candidate_adapter.py")
    write_json(target / "method.json", method)
    write_json(target / "recipe.json", recipe)
    driver = instrument(target, "timing" if recipe.get("kind") == "timing" else recipe["workload"],
                        isinstance(recipe.get("checkpoint_policy"), dict) and recipe["checkpoint_policy"].get("mode") == "rolling")
    return driver, method


def adapt_formal_runtime(runtime, recipe, method):
    """Carry explicit historical bridges into the shared plan, never the method source."""
    records = []
    if method["parameters"].get("compressor") == "powersgd":
        path = runtime / "comm_hooks/powerSGD_hook.py"
        before = file_hash(path)
        text = path.read_text()
        if "excluded_parameter_ids" not in (runtime / "candidate_adapter.py").read_text():
            raise ValueError("PowerSGD requires an explicit reviewed dense-exclusion adapter")
        if "and id(parameter) not in state.excluded_parameter_ids:" not in text:
            anchor = "for tensor in tensors:\n        matrix = tensor.view(tensor.shape[0], -1)"
            condition = "        if compress_test[0]:"
            if text.count(anchor) != 1 or text.count(condition) != 1:
                raise ValueError("PowerSGD dense-exclusion bridge anchors changed")
            text = text.replace(anchor, "for tensor, parameter in zip(tensors, bucket.parameters()):\n        matrix = tensor.view(tensor.shape[0], -1)")
            text = text.replace(condition, "        if compress_test[0] and id(parameter) not in state.excluded_parameter_ids:")
            compile(text, str(path), "exec")
            path.write_text(text)
        records.append({"file": str(path.relative_to(runtime)), "before": before, "after": file_hash(path),
                        "reason": "Apply the adapter's named dense exclusions, as in the completed PowerSGD validation."})
    if recipe.get("model_architecture") == "ResNet50":
        path = runtime / "pytorch-cifar/train_ddp.py"
        before = file_hash(path)
        text = path.read_text()
        substitutions = {"from models.resnet import BasicBlock, ResNet": "from models.resnet import Bottleneck, ResNet",
                         "ResNet(BasicBlock, [2, 2, 2, 2],": "ResNet(Bottleneck, [3, 4, 6, 3],",
                         "model='ResNet18'": "model='ResNet50'"}
        for old, new in substitutions.items():
            if text.count(old) != 1:
                raise ValueError("ResNet50 architecture bridge anchor changed")
            text = text.replace(old, new)
        compile(text, str(path), "exec")
        path.write_text(text)
        records.append({"file": str(path.relative_to(runtime)), "before": before, "after": file_hash(path),
                        "reason": "Explicit common ResNet50 architecture: Bottleneck [3,4,6,3], as requested by recipe."})
    if records:
        write_json(runtime / "formal_adaptations.json", records)
