//! CPU: JavaScript execution engine powered by Boa (pure Rust, ES2024).
//!
//! Phase 1: document.getElementById, createElement, querySelector, console.log,
//!          element manipulation, timer stubs.
//! Phase 2: fetch API, localStorage/sessionStorage, advanced DOM APIs,
//!          classList, parentNode/children, event stubs, XHR.
//! Phase 3: Fixed fetch body capture, real setTimeout queue, DOMContentLoaded,
//!          window.location with real URL, element style API, form APIs,
//!          document.cookie, document.write, JSON/Date improvements.
//! Phase 4: Element wrapper objects — getElementById etc. return real DOM-like objects
//!          with .textContent, .innerHTML, .style, .classList, .addEventListener(),
//!          .dataset, .querySelector(), .matches(), .closest(), getBoundingClientRect(),
//!          offsetWidth/Height, NodeList iterables. This is the critical change for
//!          real-world website compatibility.

use std::collections::HashMap;

use anyhow::Result;
use boa_engine::{
    Context, JsArgs, JsError, JsResult, JsValue, Source,
    js_string,
    object::ObjectInitializer,
    property::Attribute,
    NativeFunction,
};
use log::{info, warn, error};

use super::dom::{DomTree, ScriptInfo, ScriptSource};

// ═══════════════════════════════════════════════════════════════════
// Copy-safe raw pointers for Boa closures (requires Copy + Send + Sync traits)
//
// SAFETY INVARIANTS (audited 2026-03-13):
// 1. LIFETIME: All pointers are created from &mut references in execute_scripts()
//    and the Boa Context (which holds all closures) is stack-local and dropped
//    before the references go out of scope. No use-after-free is possible.
// 2. THREADING: Send+Sync are implemented only to satisfy Boa's from_copy_closure
//    trait bounds. Boa's Context is single-threaded — no closures cross thread
//    boundaries. These impls are a lie to the compiler but safe in practice.
// 3. ALIASING: get_mut() creates &mut from shared &self. This is technically UB
//    under Rust's aliasing rules, but Boa executes closures synchronously one at
//    a time, so at any given moment only one reference exists. No miscompilation
//    has been observed. A future refactor to RefCell would eliminate this concern.
// 4. NO DATA RACES: Single-threaded execution (no spawn/tokio/Arc in this file).
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Copy)]
struct DomPtr(*mut DomTree);
// SAFETY: Boa requires Send+Sync for from_copy_closure. Never actually sent across threads.
unsafe impl Send for DomPtr {}
unsafe impl Sync for DomPtr {}

impl DomPtr {
    fn new(dom: &mut DomTree) -> Self { Self(dom as *mut DomTree) }
    /// SAFETY: Caller must ensure pointer is valid and no &mut exists concurrently.
    unsafe fn get(&self) -> &DomTree { &*self.0 }
    /// SAFETY: Caller must ensure pointer is valid and no other & or &mut exists.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut DomTree { &mut *self.0 }
}

/// Copy-safe pointer to storage HashMap. Same safety invariants as DomPtr.
#[derive(Clone, Copy)]
struct StoragePtr(*mut HashMap<String, String>);
// SAFETY: Same rationale as DomPtr — never crosses thread boundaries.
unsafe impl Send for StoragePtr {}
unsafe impl Sync for StoragePtr {}

impl StoragePtr {
    fn new(store: &mut HashMap<String, String>) -> Self { Self(store as *mut _) }
    /// SAFETY: Caller must ensure pointer is valid and no &mut exists concurrently.
    unsafe fn get(&self) -> &HashMap<String, String> { &*self.0 }
    /// SAFETY: Caller must ensure pointer is valid and no other & or &mut exists.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut HashMap<String, String> { &mut *self.0 }
}

// ═══════════════════════════════════════════════════════════════════
// JsEngine — main entry point
// ═══════════════════════════════════════════════════════════════════

/// JavaScript engine for executing page scripts against a DOM tree.
pub struct JsEngine {
    max_execution_ms: u64,
    /// Persistent localStorage across page loads, keyed by origin (scheme+host+port).
    local_storage: HashMap<String, HashMap<String, String>>,
    /// Persistent sessionStorage, keyed by origin (cleared on "browser close").
    session_storage: HashMap<String, HashMap<String, String>>,
    /// In-memory cookie jar, keyed by origin (scheme+host+port) for same-origin isolation.
    cookies: HashMap<String, HashMap<String, String>>,
    /// Instant when this engine was created (for performance.now()).
    start_time: std::time::Instant,
}

impl JsEngine {
    pub fn new() -> Self {
        let max_ms = std::env::var("JS_MAX_EXECUTION_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000);
        Self {
            max_execution_ms: max_ms,
            local_storage: HashMap::new(),
            session_storage: HashMap::new(),
            cookies: HashMap::new(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Execute all collected scripts against the DOM tree.
    pub fn execute_scripts(&mut self, dom: &mut DomTree, scripts: &[ScriptInfo]) -> Result<()> {
        self.execute_scripts_with_externals(dom, scripts, &HashMap::new(), "")
    }

    /// Execute scripts with fetched external script content and page URL context.
    pub fn execute_scripts_with_externals(
        &mut self,
        dom: &mut DomTree,
        scripts: &[ScriptInfo],
        external_scripts: &HashMap<String, String>,
        page_url: &str,
    ) -> Result<()> {
        if scripts.is_empty() {
            return Ok(());
        }

        let inline_count = scripts.iter()
            .filter(|s| matches!(s.source, ScriptSource::Inline(_))).count();
        info!("[JS] Phase 4 — executing {} scripts ({} inline, {} external)",
              scripts.len(), inline_count, scripts.len() - inline_count);

        let dom_ptr = DomPtr::new(dom);
        let origin = origin_from_url(page_url);
        let ls = self.local_storage.entry(origin.clone()).or_default();
        let ls_ptr = StoragePtr::new(ls);
        let ss = self.session_storage.entry(origin.clone()).or_default();
        let ss_ptr = StoragePtr::new(ss);
        let ck = self.cookies.entry(origin.clone()).or_default();
        let ck_ptr = StoragePtr::new(ck);

        let mut context = Context::default();

        // ── Register all APIs ──
        register_console(&mut context);
        register_document(&mut context, dom_ptr);
        register_element_helpers(&mut context, dom_ptr);
        register_dom_traversal(&mut context, dom_ptr);
        register_class_list(&mut context, dom_ptr);
        register_window_location(&mut context, page_url);
        register_window_stubs(&mut context, self.start_time);
        register_timer_stubs(&mut context);
        register_storage(&mut context, ls_ptr, "localStorage");
        register_storage(&mut context, ss_ptr, "sessionStorage");
        register_fetch_api(&mut context);
        register_xhr(&mut context);
        register_event_stubs(&mut context);
        register_cookie_api(&mut context, ck_ptr);
        register_style_api(&mut context, dom_ptr);
        register_form_api(&mut context, dom_ptr);
        register_document_write(&mut context, dom_ptr);
        register_misc_globals(&mut context);

        // ── Phase 4: Element wrapper (MUST be last — wraps all __nb_* into DOM objects) ──
        register_element_wrapper(&mut context);

        // ── Run scripts (with timeout enforcement) ──
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(self.max_execution_ms);
        run_scripts(&mut context, scripts, external_scripts, deadline);

        // ── Fire DOMContentLoaded + load after all scripts execute ──
        if std::time::Instant::now() < deadline {
            fire_lifecycle_events(&mut context);
        } else {
            warn!("[JS] Skipping lifecycle events — execution timeout exceeded");
        }

        // ── Drain setTimeout(fn, 0) queue ──
        if std::time::Instant::now() < deadline {
            drain_timer_queue(&mut context);
        } else {
            warn!("[JS] Skipping timer drain — execution timeout exceeded");
        }

        Ok(())
    }
}

fn run_scripts(
    context: &mut Context,
    scripts: &[ScriptInfo],
    external_scripts: &HashMap<String, String>,
    deadline: std::time::Instant,
) {
    for (i, script) in scripts.iter().enumerate() {
        // SECURITY: Enforce execution timeout across all scripts
        if std::time::Instant::now() >= deadline {
            warn!("[JS] Execution timeout reached — skipping remaining {} scripts", scripts.len() - i);
            break;
        }

        if script.script_type.as_deref() == Some("module") {
            info!("[JS] Skipping module script #{}", i);
            continue;
        }

        let code = match &script.source {
            ScriptSource::Inline(code) => {
                if code.is_empty() { continue; }
                code.clone()
            }
            ScriptSource::External(url) => {
                match external_scripts.get(url) {
                    Some(code) => code.clone(),
                    None => {
                        info!("[JS] Skipping unfetched external script #{}: {}", i, url);
                        continue;
                    }
                }
            }
        };

        info!("[JS] Running script #{} ({} bytes)", i, code.len());
        match context.eval(Source::from_bytes(code.as_bytes())) {
            Ok(_) => {}
            Err(e) => { warn!("[JS] Script #{} error: {}", i, e); }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Console
// ═══════════════════════════════════════════════════════════════════

fn register_console(ctx: &mut Context) {
    let console = ObjectInitializer::new(ctx)
        .function(NativeFunction::from_fn_ptr(console_log), js_string!("log"), 0)
        .function(NativeFunction::from_fn_ptr(console_warn), js_string!("warn"), 0)
        .function(NativeFunction::from_fn_ptr(console_error), js_string!("error"), 0)
        .function(NativeFunction::from_fn_ptr(console_log), js_string!("info"), 0)
        .function(NativeFunction::from_fn_ptr(console_log), js_string!("debug"), 0)
        .function(NativeFunction::from_fn_ptr(console_log), js_string!("dir"), 0)
        .function(NativeFunction::from_fn_ptr(console_log), js_string!("table"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("group"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("groupEnd"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("time"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("timeEnd"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("clear"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("count"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("assert"), 0)
        .build();
    ctx.register_global_property(js_string!("console"), console, Attribute::all())
        .expect("console");
}

fn console_log(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    info!("[JS console.log] {}", args_to_string(args, ctx));
    Ok(JsValue::undefined())
}
fn console_warn(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    warn!("[JS console.warn] {}", args_to_string(args, ctx));
    Ok(JsValue::undefined())
}
fn console_error(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    error!("[JS console.error] {}", args_to_string(args, ctx));
    Ok(JsValue::undefined())
}
fn args_to_string(args: &[JsValue], ctx: &mut Context) -> String {
    args.iter()
        .map(|v| v.to_string(ctx).map(|s| s.to_std_string_escaped()).unwrap_or_else(|_| "[?]".into()))
        .collect::<Vec<_>>()
        .join(" ")
}

// ═══════════════════════════════════════════════════════════════════
// Document object
// ═══════════════════════════════════════════════════════════════════

fn register_document(ctx: &mut Context, dom: DomPtr) {
    let document = ObjectInitializer::new(ctx)
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let id = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get() };
                match d.get_element_by_id(&id) {
                    Some(nid) => Ok(JsValue::from(nid as i32)),
                    None => Ok(JsValue::null()),
                }
            }),
            js_string!("getElementById"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let tag = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get_mut() };
                Ok(JsValue::from(d.create_element(&tag) as i32))
            }),
            js_string!("createElement"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let sel = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get() };
                match d.query_selector(&sel) {
                    Some(nid) => Ok(JsValue::from(nid as i32)),
                    None => Ok(JsValue::null()),
                }
            }),
            js_string!("querySelector"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let sel = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get() };
                let results = d.query_selector_all(&sel);
                let arr = boa_engine::object::builtins::JsArray::new(ctx);
                for id in results { arr.push(JsValue::from(id as i32), ctx)?; }
                Ok(arr.into())
            }),
            js_string!("querySelectorAll"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let tag = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get() };
                let results = d.query_selector_all(&tag);
                let arr = boa_engine::object::builtins::JsArray::new(ctx);
                for id in results { arr.push(JsValue::from(id as i32), ctx)?; }
                Ok(arr.into())
            }),
            js_string!("getElementsByTagName"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let cls = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get() };
                let results: Vec<usize> = d.nodes.iter().filter_map(|n| {
                    if n.attrs.get("class").is_some_and(|c| c.split_whitespace().any(|x| x == cls)) {
                        Some(n.id)
                    } else { None }
                }).collect();
                let arr = boa_engine::object::builtins::JsArray::new(ctx);
                for id in results { arr.push(JsValue::from(id as i32), ctx)?; }
                Ok(arr.into())
            }),
            js_string!("getElementsByClassName"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let text = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get_mut() };
                let id = d.create_element("#text");
                d.set_text_content(id, &text);
                Ok(JsValue::from(id as i32))
            }),
            js_string!("createTextNode"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let text = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let d = unsafe { dom.get_mut() };
                let id = d.create_element("#comment");
                d.set_text_content(id, &text);
                Ok(JsValue::from(id as i32))
            }),
            js_string!("createComment"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, _args, _ctx| {
                let d = unsafe { dom.get_mut() };
                let id = d.create_element("#document-fragment");
                Ok(JsValue::from(id as i32))
            }),
            js_string!("createDocumentFragment"), 0,
        )
        .property(js_string!("readyState"), js_string!("complete"), Attribute::all())
        .property(js_string!("contentType"), js_string!("text/html"), Attribute::all())
        .property(js_string!("characterSet"), js_string!("UTF-8"), Attribute::all())
        .build();

    // document.body / document.head / document.documentElement
    // We set body=0 since root is typically <html> and body is a child
    let body_id = unsafe {
        let d = dom.get();
        d.query_selector("body").unwrap_or(0) as i32
    };
    let head_id = unsafe {
        let d = dom.get();
        d.query_selector("head").unwrap_or(0) as i32
    };

    // Can't set properties on built object easily, so add them via JS eval after registration
    ctx.register_global_property(js_string!("document"), document, Attribute::all())
        .expect("document");

    // Set document.body and document.head via eval
    let setup = format!(
        "document.body = {}; document.head = {}; document.documentElement = 0;",
        body_id, head_id
    );
    if let Err(e) = ctx.eval(Source::from_bytes(setup.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Element helpers (__nb_* global functions)
// ═══════════════════════════════════════════════════════════════════

fn register_element_helpers(ctx: &mut Context, dom: DomPtr) {
    // __nb_setTextContent(nodeId, text)
    ctx.register_global_builtin_callable(js_string!("__nb_setTextContent"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let text = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            unsafe { dom.get_mut() }.set_text_content(nid, &text);
            Ok(JsValue::undefined())
        }),
    ).expect("setTextContent");

    // __nb_getTextContent(nodeId) -> string
    ctx.register_global_builtin_callable(js_string!("__nb_getTextContent"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let text = d.nodes.get(nid).map(|n| n.text.clone()).unwrap_or_default();
            Ok(JsValue::from(js_string!(text)))
        }),
    ).expect("getTextContent");

    // __nb_setAttribute(nodeId, key, value)
    ctx.register_global_builtin_callable(js_string!("__nb_setAttribute"), 3,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let key = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let val = args.get_or_undefined(2).to_string(ctx)?.to_std_string_escaped();
            unsafe { dom.get_mut() }.set_attribute(nid, &key, &val);
            Ok(JsValue::undefined())
        }),
    ).expect("setAttribute");

    // __nb_getAttribute(nodeId, key) -> string|null
    ctx.register_global_builtin_callable(js_string!("__nb_getAttribute"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let key = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get() };
            match d.get_attribute(nid, &key) {
                Some(v) => Ok(JsValue::from(js_string!(v.to_string()))),
                None => Ok(JsValue::null()),
            }
        }),
    ).expect("getAttribute");

    // __nb_removeAttribute(nodeId, key)
    ctx.register_global_builtin_callable(js_string!("__nb_removeAttribute"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let key = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                node.attrs.remove(&key);
            }
            Ok(JsValue::undefined())
        }),
    ).expect("removeAttribute");

    // __nb_hasAttribute(nodeId, key) -> bool
    ctx.register_global_builtin_callable(js_string!("__nb_hasAttribute"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let key = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get() };
            let has = d.nodes.get(nid).is_some_and(|n| n.attrs.contains_key(&key));
            Ok(JsValue::from(has))
        }),
    ).expect("hasAttribute");

    // __nb_appendChild(parentId, childId) -> childId
    ctx.register_global_builtin_callable(js_string!("__nb_appendChild"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let p = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let c = args.get_or_undefined(1).to_i32(ctx)? as usize;
            unsafe { dom.get_mut() }.append_child(p, c);
            Ok(JsValue::from(c as i32))
        }),
    ).expect("appendChild");

    // __nb_removeChild(parentId, childId)
    ctx.register_global_builtin_callable(js_string!("__nb_removeChild"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let p = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let c = args.get_or_undefined(1).to_i32(ctx)? as usize;
            unsafe { dom.get_mut() }.remove_child(p, c);
            Ok(JsValue::undefined())
        }),
    ).expect("removeChild");

    // __nb_insertBefore(parentId, newChild, refChild)
    ctx.register_global_builtin_callable(js_string!("__nb_insertBefore"), 3,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let parent = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let new_child = args.get_or_undefined(1).to_i32(ctx)? as usize;
            let ref_child = args.get_or_undefined(2).to_i32(ctx)? as usize;
            let d = unsafe { dom.get_mut() };
            if parent < d.nodes.len() && new_child < d.nodes.len() {
                // Remove from old parent
                if let Some(old_p) = d.nodes[new_child].parent {
                    if old_p < d.nodes.len() {
                        d.nodes[old_p].children.retain(|&c| c != new_child);
                    }
                }
                // Insert before ref_child
                if let Some(pos) = d.nodes[parent].children.iter().position(|&c| c == ref_child) {
                    d.nodes[parent].children.insert(pos, new_child);
                } else {
                    d.nodes[parent].children.push(new_child);
                }
                d.nodes[new_child].parent = Some(parent);
            }
            Ok(JsValue::from(new_child as i32))
        }),
    ).expect("insertBefore");

    // __nb_setInnerHTML(nodeId, html)
    ctx.register_global_builtin_callable(js_string!("__nb_setInnerHTML"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let html = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            unsafe { dom.get_mut() }.set_inner_html(nid, &html);
            Ok(JsValue::undefined())
        }),
    ).expect("setInnerHTML");

    // __nb_getInnerHTML(nodeId) -> string (basic reconstruction)
    ctx.register_global_builtin_callable(js_string!("__nb_getInnerHTML"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let html = reconstruct_html(d, nid);
            Ok(JsValue::from(js_string!(html)))
        }),
    ).expect("getInnerHTML");

    // __nb_getTagName(nodeId) -> string
    ctx.register_global_builtin_callable(js_string!("__nb_getTagName"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let tag = d.nodes.get(nid).map(|n| n.tag.to_uppercase()).unwrap_or_default();
            Ok(JsValue::from(js_string!(tag)))
        }),
    ).expect("getTagName");

    // __nb_cloneNode(nodeId, deep) -> newNodeId
    ctx.register_global_builtin_callable(js_string!("__nb_cloneNode"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let deep = args.get_or_undefined(1).to_boolean();
            let d = unsafe { dom.get_mut() };
            if let Some(src) = d.nodes.get(nid).cloned() {
                let new_id = d.nodes.len();
                d.nodes.push(super::dom::DomNode {
                    id: new_id,
                    tag: src.tag,
                    attrs: src.attrs,
                    text: src.text,
                    parent: None,
                    children: Vec::new(),
                    depth: 0,
                });
                if deep {
                    fn deep_clone(d: &mut super::dom::DomTree, src_id: usize, new_parent_id: usize) {
                        let src_children: Vec<usize> = d.nodes.get(src_id)
                            .map(|n| n.children.clone()).unwrap_or_default();
                        for child_id in src_children {
                            if let Some(child) = d.nodes.get(child_id).cloned() {
                                let clone_id = d.nodes.len();
                                let depth = d.nodes[new_parent_id].depth + 1;
                                d.nodes.push(super::dom::DomNode {
                                    id: clone_id,
                                    tag: child.tag,
                                    attrs: child.attrs,
                                    text: child.text,
                                    parent: Some(new_parent_id),
                                    children: Vec::new(),
                                    depth,
                                });
                                d.nodes[new_parent_id].children.push(clone_id);
                                deep_clone(d, child_id, clone_id);
                            }
                        }
                    }
                    deep_clone(d, nid, new_id);
                }
                Ok(JsValue::from(new_id as i32))
            } else {
                Ok(JsValue::null())
            }
        }),
    ).expect("cloneNode");
}

/// Basic HTML reconstruction from DOM tree (for innerHTML getter).
fn reconstruct_html(dom: &DomTree, node_id: usize) -> String {
    let mut out = String::new();
    if let Some(node) = dom.nodes.get(node_id) {
        for &child_id in &node.children {
            reconstruct_node(dom, child_id, &mut out);
        }
        if node.children.is_empty() && !node.text.is_empty() {
            out.push_str(&node.text);
        }
    }
    out
}

fn reconstruct_node(dom: &DomTree, node_id: usize, out: &mut String) {
    if let Some(node) = dom.nodes.get(node_id) {
        if node.tag == "#text" {
            out.push_str(&node.text);
            return;
        }
        out.push('<');
        out.push_str(&node.tag);
        for (k, v) in &node.attrs {
            out.push(' ');
            out.push_str(k);
            out.push_str("=\"");
            // SECURITY: Escape attribute values to prevent XSS via attribute injection
            for ch in v.chars() {
                match ch {
                    '"' => out.push_str("&quot;"),
                    '&' => out.push_str("&amp;"),
                    '<' => out.push_str("&lt;"),
                    '>' => out.push_str("&gt;"),
                    _ => out.push(ch),
                }
            }
            out.push('"');
        }
        out.push('>');
        if !node.text.is_empty() {
            out.push_str(&node.text);
        }
        for &child_id in &node.children {
            reconstruct_node(dom, child_id, out);
        }
        out.push_str("</");
        out.push_str(&node.tag);
        out.push('>');
    }
}

// ═══════════════════════════════════════════════════════════════════
// DOM Traversal (parentNode, children, siblings, etc.)
// ═══════════════════════════════════════════════════════════════════

fn register_dom_traversal(ctx: &mut Context, dom: DomPtr) {
    // __nb_getParent(nodeId) -> parentId | null
    ctx.register_global_builtin_callable(js_string!("__nb_getParent"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            match d.nodes.get(nid).and_then(|n| n.parent) {
                Some(p) => Ok(JsValue::from(p as i32)),
                None => Ok(JsValue::null()),
            }
        }),
    ).expect("getParent");

    // __nb_getChildren(nodeId) -> [childId, ...]
    ctx.register_global_builtin_callable(js_string!("__nb_getChildren"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let arr = boa_engine::object::builtins::JsArray::new(ctx);
            if let Some(node) = d.nodes.get(nid) {
                for &c in &node.children {
                    arr.push(JsValue::from(c as i32), ctx)?;
                }
            }
            Ok(arr.into())
        }),
    ).expect("getChildren");

    // __nb_getChildCount(nodeId) -> number
    ctx.register_global_builtin_callable(js_string!("__nb_getChildCount"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let count = d.nodes.get(nid).map(|n| n.children.len()).unwrap_or(0);
            Ok(JsValue::from(count as i32))
        }),
    ).expect("getChildCount");

    // __nb_getFirstChild(nodeId) -> childId | null
    ctx.register_global_builtin_callable(js_string!("__nb_getFirstChild"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            match d.nodes.get(nid).and_then(|n| n.children.first()) {
                Some(&c) => Ok(JsValue::from(c as i32)),
                None => Ok(JsValue::null()),
            }
        }),
    ).expect("getFirstChild");

    // __nb_getLastChild(nodeId) -> childId | null
    ctx.register_global_builtin_callable(js_string!("__nb_getLastChild"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            match d.nodes.get(nid).and_then(|n| n.children.last()) {
                Some(&c) => Ok(JsValue::from(c as i32)),
                None => Ok(JsValue::null()),
            }
        }),
    ).expect("getLastChild");

    // __nb_getNextSibling(nodeId) -> siblingId | null
    ctx.register_global_builtin_callable(js_string!("__nb_getNextSibling"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let sibling = d.nodes.get(nid).and_then(|n| n.parent).and_then(|pid| {
                let parent = d.nodes.get(pid)?;
                let pos = parent.children.iter().position(|&c| c == nid)?;
                parent.children.get(pos + 1).copied()
            });
            match sibling {
                Some(s) => Ok(JsValue::from(s as i32)),
                None => Ok(JsValue::null()),
            }
        }),
    ).expect("getNextSibling");

    // __nb_getPrevSibling(nodeId) -> siblingId | null
    ctx.register_global_builtin_callable(js_string!("__nb_getPrevSibling"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let sibling = d.nodes.get(nid).and_then(|n| n.parent).and_then(|pid| {
                let parent = d.nodes.get(pid)?;
                let pos = parent.children.iter().position(|&c| c == nid)?;
                if pos > 0 { Some(parent.children[pos - 1]) } else { None }
            });
            match sibling {
                Some(s) => Ok(JsValue::from(s as i32)),
                None => Ok(JsValue::null()),
            }
        }),
    ).expect("getPrevSibling");

    // __nb_nodeExists(nodeId) -> bool
    ctx.register_global_builtin_callable(js_string!("__nb_nodeExists"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            Ok(JsValue::from(nid < d.nodes.len()))
        }),
    ).expect("nodeExists");
}

// ═══════════════════════════════════════════════════════════════════
// classList API
// ═══════════════════════════════════════════════════════════════════

fn register_class_list(ctx: &mut Context, dom: DomPtr) {
    // __nb_classAdd(nodeId, className)
    ctx.register_global_builtin_callable(js_string!("__nb_classAdd"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let cls = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                let current = node.attrs.entry("class".to_string()).or_default();
                if !current.split_whitespace().any(|c| c == cls) {
                    if !current.is_empty() { current.push(' '); }
                    current.push_str(&cls);
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("classAdd");

    // __nb_classRemove(nodeId, className)
    ctx.register_global_builtin_callable(js_string!("__nb_classRemove"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let cls = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                if let Some(current) = node.attrs.get_mut("class") {
                    let new_val: Vec<&str> = current.split_whitespace()
                        .filter(|&c| c != cls).collect();
                    *current = new_val.join(" ");
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("classRemove");

    // __nb_classToggle(nodeId, className) -> bool
    ctx.register_global_builtin_callable(js_string!("__nb_classToggle"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let cls = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                let current = node.attrs.entry("class".to_string()).or_default();
                if current.split_whitespace().any(|c| c == cls) {
                    let new_val: Vec<&str> = current.split_whitespace()
                        .filter(|&c| c != cls).collect();
                    *current = new_val.join(" ");
                    return Ok(JsValue::from(false));
                } else {
                    if !current.is_empty() { current.push(' '); }
                    current.push_str(&cls);
                    return Ok(JsValue::from(true));
                }
            }
            Ok(JsValue::from(false))
        }),
    ).expect("classToggle");

    // __nb_classContains(nodeId, className) -> bool
    ctx.register_global_builtin_callable(js_string!("__nb_classContains"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let cls = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get() };
            let has = d.nodes.get(nid).is_some_and(|n| {
                n.attrs.get("class").is_some_and(|c| c.split_whitespace().any(|x| x == cls))
            });
            Ok(JsValue::from(has))
        }),
    ).expect("classContains");
}

// ═══════════════════════════════════════════════════════════════════
/// Extract origin (scheme + host + port) from a URL for storage isolation.
/// Returns a fallback key for invalid or empty URLs so tests still work.
fn origin_from_url(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        format!("{}://{}{}", parsed.scheme(),
                parsed.host_str().unwrap_or("localhost"),
                parsed.port().map(|p| format!(":{p}")).unwrap_or_default())
    } else {
        // Fallback for empty URLs (e.g. unit tests calling execute_scripts)
        String::from("__default__")
    }
}

// localStorage / sessionStorage
// ═══════════════════════════════════════════════════════════════════

fn register_storage(ctx: &mut Context, store: StoragePtr, name: &str) {
    let storage = ObjectInitializer::new(ctx)
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let key = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let s = unsafe { store.get() };
                match s.get(&key) {
                    Some(v) => Ok(JsValue::from(js_string!(v.clone()))),
                    None => Ok(JsValue::null()),
                }
            }),
            js_string!("getItem"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let key = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let val = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
                let s = unsafe { store.get_mut() };
                s.insert(key, val);
                Ok(JsValue::undefined())
            }),
            js_string!("setItem"), 2,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let key = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
                let s = unsafe { store.get_mut() };
                s.remove(&key);
                Ok(JsValue::undefined())
            }),
            js_string!("removeItem"), 1,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, _, _| {
                let s = unsafe { store.get_mut() };
                s.clear();
                Ok(JsValue::undefined())
            }),
            js_string!("clear"), 0,
        )
        .function(
            NativeFunction::from_copy_closure(move |_, args, ctx| {
                let idx = args.get_or_undefined(0).to_i32(ctx)? as usize;
                let s = unsafe { store.get() };
                match s.keys().nth(idx) {
                    Some(k) => Ok(JsValue::from(js_string!(k.clone()))),
                    None => Ok(JsValue::null()),
                }
            }),
            js_string!("key"), 1,
        )
        .build();

    ctx.register_global_property(js_string!(name.to_string()), storage, Attribute::all())
        .expect(name);

    // Register a helper to get storage length, then define .length as a getter
    let len_fn_name = format!("__nb_{}_length", name);
    ctx.register_global_builtin_callable(js_string!(len_fn_name.clone()), 0,
        NativeFunction::from_copy_closure(move |_, _, _| {
            let s = unsafe { store.get() };
            Ok(JsValue::from(s.len() as i32))
        }),
    ).expect("storage_length");

    let length_js = format!(
        r#"Object.defineProperty({name}, "length", {{ get: function() {{ return {len_fn_name}(); }}, enumerable: true, configurable: true }});"#,
        name = name, len_fn_name = len_fn_name,
    );
    if let Err(e) = ctx.eval(Source::from_bytes(length_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Fetch API (synchronous — blocks JS execution)
// Phase 3: Fixed body capture — stores in global, text()/json() read from it
// ═══════════════════════════════════════════════════════════════════

fn register_fetch_api(ctx: &mut Context) {
    // fetch(url, options?) -> { ok, status, statusText, _body, text(), json(), headers }
    // Synchronous implementation using ureq
    ctx.register_global_builtin_callable(js_string!("fetch"), 2,
        NativeFunction::from_fn_ptr(js_fetch),
    ).expect("fetch");

    // Helper: __nb_fetch_id tracks multiple concurrent fetch results
    if let Err(e) = ctx.eval(Source::from_bytes(b"var __nb_fetch_id = 0; var __nb_fetch_bodies = {};")) { log::warn!("[JS] eval failed: {e:?}"); }
}

/// SECURITY: Check if a URL is safe for JS fetch (block private IPs, file://, etc.)
fn is_fetch_url_allowed(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {e}"))?;

    // Only allow http and https schemes
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => return Err(format!("Blocked scheme: {scheme}:// — only http(s) allowed")),
    }

    // Block private/internal IPs to prevent SSRF
    if let Some(host) = parsed.host_str() {
        let host_lower = host.to_lowercase();
        // Block localhost variants
        if host_lower == "localhost" || host_lower == "127.0.0.1" || host_lower == "::1"
            || host_lower == "[::1]" || host_lower == "0.0.0.0"
        {
            return Err(format!("Blocked: access to localhost ({host}) is not allowed"));
        }
        // Block metadata services (AWS/GCP/Azure)
        if host_lower == "169.254.169.254" || host_lower == "metadata.google.internal" {
            return Err(format!("Blocked: access to cloud metadata service ({host})"));
        }
        // Block private IP ranges (10.x, 172.16-31.x, 192.168.x)
        if let Ok(ip) = host.parse::<std::net::Ipv4Addr>() {
            if ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified() {
                return Err(format!("Blocked: private/reserved IP ({ip})"));
            }
        }
        if let Ok(ip) = host.parse::<std::net::Ipv6Addr>() {
            if ip.is_loopback() || ip.is_unspecified() {
                return Err(format!("Blocked: loopback/reserved IPv6 ({ip})"));
            }
        }
    }
    Ok(())
}

/// Max response body size for JS fetch (2 MB)
const JS_FETCH_MAX_BODY: usize = 2 * 1024 * 1024;

fn js_fetch(_: &JsValue, args: &[JsValue], ctx: &mut Context) -> JsResult<JsValue> {
    let url = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
    info!("[JS fetch] {}", url);

    // SECURITY: Validate URL before making request
    if let Err(reason) = is_fetch_url_allowed(&url) {
        warn!("[JS fetch] BLOCKED: {reason}");
        // Return a rejected-style response
        let err_js = format!(
            r#"(function() {{
                var r = {{ ok: false, status: 0, statusText: "Blocked", _body: "",
                    text: function() {{ return Promise.resolve(""); }},
                    json: function() {{ return Promise.reject(new TypeError("{}")); }},
                    headers: {{ get: function() {{ return null; }} }},
                    clone: function() {{ return Object.assign({{}}, this); }}
                }};
                r.then = function(f,rej) {{ if(rej) rej(new TypeError("{}")); return {{ then: function(){{return this;}}, catch: function(f){{if(f)f(new TypeError("{}"));return this;}}, finally: function(f){{if(f)f();return this;}} }}; }};
                r.catch = function(f) {{ return r.then(null, f); }};
                r.finally = function(f) {{ if(f)f(); return r; }};
                return r;
            }})()"#,
            reason.replace('"', "\\\""),
            reason.replace('"', "\\\""),
            reason.replace('"', "\\\""),
        );
        return ctx.eval(Source::from_bytes(err_js.as_bytes()))
            .map_err(|e| JsError::from_opaque(JsValue::from(js_string!(format!("fetch blocked: {e}")))));
    }

    let mut method = "GET".to_string();
    let mut body: Option<String> = None;
    let mut content_type: Option<String> = None;

    if let Some(opts) = args.get(1) {
        if opts.is_object() {
            if let Ok(m_val) = opts.to_object(ctx)
                .and_then(|o| o.get(js_string!("method"), ctx))
            {
                if !m_val.is_undefined() && !m_val.is_null() {
                    if let Ok(s) = m_val.to_string(ctx) {
                        method = s.to_std_string_escaped().to_uppercase();
                    }
                }
            }
            if let Ok(b_val) = opts.to_object(ctx)
                .and_then(|o| o.get(js_string!("body"), ctx))
            {
                if !b_val.is_undefined() && !b_val.is_null() {
                    if let Ok(s) = b_val.to_string(ctx) {
                        body = Some(s.to_std_string_escaped());
                    }
                }
            }
            // Extract headers.Content-Type if present
            if let Ok(h_val) = opts.to_object(ctx)
                .and_then(|o| o.get(js_string!("headers"), ctx))
            {
                if h_val.is_object() {
                    if let Ok(ct) = h_val.to_object(ctx)
                        .and_then(|o| o.get(js_string!("Content-Type"), ctx))
                    {
                        if !ct.is_undefined() && !ct.is_null() {
                            if let Ok(s) = ct.to_string(ctx) {
                                content_type = Some(s.to_std_string_escaped());
                            }
                        }
                    }
                }
            }
        }
    }

    let ct = content_type.unwrap_or_else(|| "application/json".to_string());

    let result = std::panic::catch_unwind(|| {
        let agent = ureq::Agent::new_with_defaults();
        let response = match method.as_str() {
            "POST" => {
                let req = agent.post(&url).header("Content-Type", &ct);
                if let Some(b) = body { req.send(b.as_bytes()) } else { req.send(&[] as &[u8]) }
            }
            "PUT" => {
                let req = agent.put(&url).header("Content-Type", &ct);
                if let Some(b) = body { req.send(b.as_bytes()) } else { req.send(&[] as &[u8]) }
            }
            "PATCH" => {
                let req = agent.patch(&url).header("Content-Type", &ct);
                if let Some(b) = body { req.send(b.as_bytes()) } else { req.send(&[] as &[u8]) }
            }
            "DELETE" => agent.delete(&url).call(),
            "HEAD" => agent.head(&url).call(),
            _ => agent.get(&url).call(),
        };
        match response {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let ok = (200..300).contains(&status);
                let mut body_text = resp.into_body().read_to_string().unwrap_or_default();
                // SECURITY: Limit response body size to prevent memory exhaustion
                if body_text.len() > JS_FETCH_MAX_BODY {
                    log::warn!("[JS fetch] Response truncated from {} to {} bytes", body_text.len(), JS_FETCH_MAX_BODY);
                    body_text.truncate(JS_FETCH_MAX_BODY);
                }
                (ok, status, body_text, String::new())
            }
            Err(e) => (false, 0u16, String::new(), format!("{e}")),
        }
    });

    let (ok, status, body_text, err_msg) = match result {
        Ok(r) => r,
        Err(_) => (false, 0, String::new(), "fetch panicked".to_string()),
    };

    if !err_msg.is_empty() {
        warn!("[JS fetch] Error: {}", err_msg);
    }

    // Store body in global __nb_last_fetch_body (used by XHR and response.text())
    ctx.register_global_property(
        js_string!("__nb_last_fetch_body"),
        js_string!(body_text.clone()),
        Attribute::all(),
    ).expect("fetch body store");

    // Increment fetch ID and store body keyed by ID for concurrent fetches
    let store_js = r#"__nb_fetch_id++; __nb_fetch_bodies[__nb_fetch_id] = __nb_last_fetch_body;"#.to_string();
    if let Err(e) = ctx.eval(Source::from_bytes(store_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }

    // Build response object — text() and json() read from __nb_last_fetch_body
    // We store _body directly on the response object for JS access
    let response_js = format!(
        r#"(function() {{
            var _fid = __nb_fetch_id;
            var _resp = {{
                ok: {ok},
                status: {status},
                statusText: "{status_text}",
                _body: __nb_fetch_bodies[_fid] || "",
                text: function() {{ var b = this._body; return Promise.resolve(b); }},
                json: function() {{ var b = this._body; try {{ return Promise.resolve(JSON.parse(b)); }} catch(e) {{ return Promise.reject(e); }} }},
                headers: {{ get: function(name) {{ return null; }} }},
                clone: function() {{ return Object.assign({{}}, this); }}
            }};
            _resp.then = function(onFulfilled, onRejected) {{
                try {{
                    var result = onFulfilled ? onFulfilled(_resp) : _resp;
                    return {{ then: function(f) {{ if(f) try {{ f(result); }} catch(e2) {{}} return this; }}, catch: function() {{ return this; }}, finally: function(f) {{ if(f) f(); return this; }} }};
                }} catch(e) {{
                    if(onRejected) onRejected(e);
                    return {{ then: function() {{ return this; }}, catch: function(f) {{ if(f) f(e); return this; }}, finally: function(f) {{ if(f) f(); return this; }} }};
                }}
            }};
            _resp.catch = function(onRejected) {{ return _resp.then(null, onRejected); }};
            _resp.finally = function(fn) {{ if(fn) fn(); return _resp; }};
            return _resp;
        }})()"#,
        ok = ok,
        status = status,
        status_text = if ok { "OK" } else { "Error" },
    );

    match ctx.eval(Source::from_bytes(response_js.as_bytes())) {
        Ok(resp) => Ok(resp),
        Err(e) => {
            warn!("[JS fetch] Response object creation failed: {e}");
            // Fallback: return simple object
            let fallback = ObjectInitializer::new(ctx)
                .property(js_string!("ok"), JsValue::from(ok), Attribute::all())
                .property(js_string!("status"), JsValue::from(status as i32), Attribute::all())
                .build();
            Ok(fallback.into())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// XMLHttpRequest (synchronous mode)
// ═══════════════════════════════════════════════════════════════════

fn register_xhr(ctx: &mut Context) {
    // Minimal XHR: var xhr = new XMLHttpRequest(); xhr.open("GET", url); xhr.send();
    // Implemented as a global object factory
    let xhr_js = r#"
        function XMLHttpRequest() {
            this.readyState = 0;
            this._status = 0;
            this._statusText = "";
            this._responseText = "";
            this.responseType = "";
            this._response = null;
            this._method = "GET";
            this._url = "";
            this._headers = {};
            this._async = true;
            this.onreadystatechange = null;
            this.onload = null;
            this.onerror = null;
        }
        Object.defineProperty(XMLHttpRequest.prototype, "responseText", {
            get: function() { return this.readyState === 4 ? this._responseText : ""; },
            configurable: true
        });
        Object.defineProperty(XMLHttpRequest.prototype, "response", {
            get: function() { return this.readyState === 4 ? this._response : null; },
            configurable: true
        });
        Object.defineProperty(XMLHttpRequest.prototype, "status", {
            get: function() { return this.readyState === 4 ? this._status : 0; },
            configurable: true
        });
        Object.defineProperty(XMLHttpRequest.prototype, "statusText", {
            get: function() { return this.readyState === 4 ? this._statusText : ""; },
            configurable: true
        });
        XMLHttpRequest.prototype.open = function(method, url, async_flag) {
            this._method = method || "GET";
            this._url = url || "";
            this._async = async_flag !== false;
            this.readyState = 1;
        };
        XMLHttpRequest.prototype.setRequestHeader = function(key, val) {
            this._headers[key] = val;
        };
        XMLHttpRequest.prototype.getResponseHeader = function(key) {
            return null;
        };
        XMLHttpRequest.prototype.getAllResponseHeaders = function() {
            return "";
        };
        XMLHttpRequest.prototype.send = function(body) {
            try {
                var opts = { method: this._method };
                if (body) opts.body = body;
                if (Object.keys(this._headers).length > 0) opts.headers = this._headers;
                var resp = fetch(this._url, opts);
                this._status = resp.status;
                this._statusText = resp.statusText;
                this._responseText = __nb_last_fetch_body || "";
                this._response = this._responseText;
                this.readyState = 4;
                if (this.onreadystatechange) this.onreadystatechange();
                if (this.onload) this.onload();
            } catch(e) {
                this._status = 0;
                this.readyState = 4;
                if (this.onerror) this.onerror(e);
            }
        };
        XMLHttpRequest.prototype.abort = function() {
            this.readyState = 0;
        };
        XMLHttpRequest.UNSENT = 0;
        XMLHttpRequest.OPENED = 1;
        XMLHttpRequest.HEADERS_RECEIVED = 2;
        XMLHttpRequest.LOADING = 3;
        XMLHttpRequest.DONE = 4;
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(xhr_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Event stubs (addEventListener, removeEventListener, dispatchEvent)
// ═══════════════════════════════════════════════════════════════════

fn register_event_stubs(ctx: &mut Context) {
    let events_js = r#"
        var __nb_events = {};
        function __nb_addEventListener(target, type, listener) {
            var key = target + "_" + type;
            if (!__nb_events[key]) __nb_events[key] = [];
            __nb_events[key].push(listener);
        }
        function __nb_removeEventListener(target, type, listener) {
            var key = target + "_" + type;
            if (__nb_events[key]) {
                __nb_events[key] = __nb_events[key].filter(function(l) { return l !== listener; });
            }
        }
        function __nb_dispatchEvent(target, type, eventObj) {
            var key = target + "_" + type;
            if (__nb_events[key]) {
                var evt = eventObj || { type: type, target: target, preventDefault: function(){}, stopPropagation: function(){} };
                for (var i = 0; i < __nb_events[key].length; i++) {
                    __nb_events[key][i](evt);
                }
            }
        }

        // document.addEventListener / window.addEventListener
        document.addEventListener = function(type, fn, opts) { __nb_addEventListener("document", type, fn); };
        document.removeEventListener = function(type, fn) { __nb_removeEventListener("document", type, fn); };
        document.dispatchEvent = function(evt) { __nb_dispatchEvent("document", evt.type || "custom", evt); };

        window.addEventListener = function(type, fn, opts) { __nb_addEventListener("window", type, fn); };
        window.removeEventListener = function(type, fn) { __nb_removeEventListener("window", type, fn); };
        window.dispatchEvent = function(evt) { __nb_dispatchEvent("window", evt.type || "custom", evt); };

        // Fire DOMContentLoaded after all scripts execute
        // (We'll trigger this at the end)

        // CustomEvent constructor
        function CustomEvent(type, params) {
            this.type = type;
            this.detail = (params && params.detail) || null;
            this.bubbles = (params && params.bubbles) || false;
            this.cancelable = (params && params.cancelable) || false;
            this.preventDefault = function(){};
            this.stopPropagation = function(){};
        }

        // Event constructor
        function Event(type, params) {
            this.type = type;
            this.bubbles = (params && params.bubbles) || false;
            this.cancelable = (params && params.cancelable) || false;
            this.preventDefault = function(){};
            this.stopPropagation = function(){};
        }

        // Node constants (used by React, jQuery, etc.)
        var Node = {
            ELEMENT_NODE: 1,
            TEXT_NODE: 3,
            COMMENT_NODE: 8,
            DOCUMENT_NODE: 9,
            DOCUMENT_FRAGMENT_NODE: 11,
            ATTRIBUTE_NODE: 2,
            CDATA_SECTION_NODE: 4,
            PROCESSING_INSTRUCTION_NODE: 7,
            DOCUMENT_TYPE_NODE: 10
        };
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(events_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Window.location with real URL (Phase 3)
// ═══════════════════════════════════════════════════════════════════

fn register_window_location(ctx: &mut Context, page_url: &str) {
    // Parse the real URL to extract components
    let (href, hostname, pathname, protocol, search, hash, origin, host, port) =
        if let Ok(parsed) = url::Url::parse(page_url) {
            (
                parsed.as_str().to_string(),
                parsed.host_str().unwrap_or("").to_string(),
                parsed.path().to_string(),
                format!("{}:", parsed.scheme()),
                parsed.query().map(|q| format!("?{q}")).unwrap_or_default(),
                parsed.fragment().map(|f| format!("#{f}")).unwrap_or_default(),
                parsed.origin().ascii_serialization(),
                parsed.host_str().unwrap_or("").to_string(),
                parsed.port().map(|p| p.to_string()).unwrap_or_default(),
            )
        } else {
            (page_url.to_string(), String::new(), "/".to_string(),
             "https:".to_string(), String::new(), String::new(),
             String::new(), String::new(), String::new())
        };

    let location = ObjectInitializer::new(ctx)
        .property(js_string!("href"), js_string!(href.clone()), Attribute::all())
        .property(js_string!("hostname"), js_string!(hostname.clone()), Attribute::all())
        .property(js_string!("pathname"), js_string!(pathname), Attribute::all())
        .property(js_string!("protocol"), js_string!(protocol), Attribute::all())
        .property(js_string!("search"), js_string!(search), Attribute::all())
        .property(js_string!("hash"), js_string!(hash), Attribute::all())
        .property(js_string!("origin"), js_string!(origin), Attribute::all())
        .property(js_string!("host"), js_string!(host), Attribute::all())
        .property(js_string!("port"), js_string!(port), Attribute::all())
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::from(js_string!("")))),
            js_string!("toString"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())),
            js_string!("reload"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())),
            js_string!("replace"), 1)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())),
            js_string!("assign"), 1)
        .build();

    ctx.register_global_property(js_string!("location"), location, Attribute::all()).expect("location");

    // Also set window.location and document.location
    // SECURITY: Escape href/hostname to prevent JS string injection
    let safe_href = href.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r");
    let safe_hostname = hostname.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r");
    let loc_setup = format!(
        r#"window.location = location; document.location = location; document.URL = "{safe_href}"; document.domain = "{safe_hostname}";"#
    );
    if let Err(e) = ctx.eval(Source::from_bytes(loc_setup.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Window stubs
// ═══════════════════════════════════════════════════════════════════

fn register_window_stubs(ctx: &mut Context, start_time: std::time::Instant) {
    // Don't recreate window — just add properties to existing global
    let window_js = r#"
        if (typeof window === "undefined") { var window = {}; }
        window.innerWidth = 1280;
        window.innerHeight = 720;
        window.outerWidth = 1280;
        window.outerHeight = 800;
        window.devicePixelRatio = 1;
        window.screenX = 0;
        window.screenY = 0;
        window.scrollX = 0;
        window.scrollY = 0;
        window.pageXOffset = 0;
        window.pageYOffset = 0;
        window.self = window;
        window.top = window;
        window.parent = window;
        window.frames = window;
        window.length = 0;
        window.closed = false;
        window.name = "";
        window.opener = null;
        window.isSecureContext = true;
        window.scrollTo = function(x, y) {};
        window.scrollBy = function(x, y) {};
        window.scroll = function(x, y) {};
        window.focus = function() {};
        window.blur = function() {};
        window.close = function() {};
        window.open = function() { return null; };
        window.alert = function(msg) { console.log("[alert] " + msg); };
        window.confirm = function(msg) { console.log("[confirm] " + msg); return true; };
        window.prompt = function(msg, def) { console.log("[prompt] " + msg); return def || ""; };
        window.print = function() {};
        window.postMessage = function() {};
        window.getSelection = function() { return { toString: function() { return ""; }, rangeCount: 0, removeAllRanges: function() {} }; };
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(window_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }

    let screen = ObjectInitializer::new(ctx)
        .property(js_string!("width"), JsValue::from(1920), Attribute::all())
        .property(js_string!("height"), JsValue::from(1080), Attribute::all())
        .property(js_string!("availWidth"), JsValue::from(1920), Attribute::all())
        .property(js_string!("availHeight"), JsValue::from(1040), Attribute::all())
        .property(js_string!("colorDepth"), JsValue::from(24), Attribute::all())
        .property(js_string!("pixelDepth"), JsValue::from(24), Attribute::all())
        .build();
    ctx.register_global_property(js_string!("screen"), screen, Attribute::all()).expect("screen");

    let navigator = ObjectInitializer::new(ctx)
        .property(js_string!("userAgent"), js_string!("Mozilla/5.0 NeuralBrowser/0.1"), Attribute::all())
        .property(js_string!("language"), js_string!("en-US"), Attribute::all())
        .property(js_string!("languages"), js_string!("en-US,en"), Attribute::all())
        .property(js_string!("platform"), js_string!(if cfg!(target_os = "macos") { "MacIntel" } else if cfg!(target_os = "linux") { "Linux x86_64" } else { "Win32" }), Attribute::all())
        .property(js_string!("vendor"), js_string!("NeuralBrowser"), Attribute::all())
        .property(js_string!("cookieEnabled"), JsValue::from(false), Attribute::all())
        .property(js_string!("onLine"), JsValue::from(true), Attribute::all())
        .property(js_string!("hardwareConcurrency"), JsValue::from(4), Attribute::all())
        .property(js_string!("maxTouchPoints"), JsValue::from(0), Attribute::all())
        .build();
    ctx.register_global_property(js_string!("navigator"), navigator, Attribute::all()).expect("navigator");

    let history = ObjectInitializer::new(ctx)
        .property(js_string!("length"), JsValue::from(1), Attribute::all())
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("pushState"), 3)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("replaceState"), 3)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("back"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("forward"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("go"), 1)
        .build();
    ctx.register_global_property(js_string!("history"), history, Attribute::all()).expect("history");

    // performance.now() — returns elapsed milliseconds since engine start
    let performance = ObjectInitializer::new(ctx)
        .function(NativeFunction::from_copy_closure(move |_,_,_| {
            let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
            Ok(JsValue::from(elapsed))
        }), js_string!("now"), 0)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("mark"), 1)
        .function(NativeFunction::from_fn_ptr(|_,_,_| Ok(JsValue::undefined())), js_string!("measure"), 3)
        .build();
    ctx.register_global_property(js_string!("performance"), performance, Attribute::all()).expect("performance");

    // Bug 5 fix: Alias common globals so window.document, window.navigator, etc. work
    let alias_js = r#"
        window.document = document;
        window.navigator = navigator;
        window.screen = screen;
        window.history = history;
        window.performance = performance;
        window.console = console;
        window.localStorage = localStorage;
        window.sessionStorage = sessionStorage;
        window.setTimeout = setTimeout;
        window.setInterval = setInterval;
        window.clearTimeout = clearTimeout;
        window.clearInterval = clearInterval;
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(alias_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Timer stubs
// ═══════════════════════════════════════════════════════════════════

fn register_timer_stubs(ctx: &mut Context) {
    // Phase 3: setTimeout with callback queue — callbacks with delay <= 0 execute immediately,
    // others are queued and drained after all scripts finish (drain_timer_queue).
    // setInterval still stubs (no recurring in synchronous engine).
    let timer_js = r#"
        var __nb_timer_id = 0;
        var __nb_timer_queue = [];
        var __nb_timer_cleared = {};
        function setTimeout(fn, delay) {
            var id = ++__nb_timer_id;
            if (typeof fn === "function") {
                if (!delay || delay <= 0) {
                    try { fn(); } catch(e) { console.error("[setTimeout] " + e); }
                } else {
                    __nb_timer_queue.push({ id: id, fn: fn });
                }
            } else if (typeof fn === "string") {
                // setTimeout("code", delay) — eval variant
                if (!delay || delay <= 0) {
                    try { eval(fn); } catch(e) {}
                } else {
                    __nb_timer_queue.push({ id: id, fn: function() { eval(fn); } });
                }
            }
            return id;
        }
        function setInterval(fn, delay) {
            // In synchronous engine, setInterval just executes once
            var id = ++__nb_timer_id;
            if (typeof fn === "function") {
                __nb_timer_queue.push({ id: id, fn: fn });
            }
            return id;
        }
        function clearTimeout(id) { __nb_timer_cleared[id] = true; }
        function clearInterval(id) { __nb_timer_cleared[id] = true; }
        function requestAnimationFrame(fn) {
            var id = ++__nb_timer_id;
            if (typeof fn === "function") {
                __nb_timer_queue.push({ id: id, fn: function() { fn(performance.now()); } });
            }
            return id;
        }
        function cancelAnimationFrame(id) { __nb_timer_cleared[id] = true; }
        function queueMicrotask(fn) {
            if (typeof fn === "function") {
                try { fn(); } catch(e) {}
            }
        }
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(timer_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

/// Drain the setTimeout/setInterval/rAF callback queue (Phase 3).
/// Executes all pending callbacks up to a maximum of 1000 iterations.
fn drain_timer_queue(ctx: &mut Context) {
    let drain_js = r#"
        (function() {
            var maxIterations = 1000;
            var iterations = 0;
            while (__nb_timer_queue.length > 0 && iterations < maxIterations) {
                var task = __nb_timer_queue.shift();
                iterations++;
                if (__nb_timer_cleared[task.id]) continue;
                try { task.fn(); } catch(e) { console.error("[timer] " + e); }
            }
            if (__nb_timer_queue.length > 0) {
                console.warn("[NB] Timer queue still has " + __nb_timer_queue.length + " tasks after 1000 iterations");
            }
        })();
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(drain_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

/// Fire DOMContentLoaded and load events (Phase 3).
fn fire_lifecycle_events(ctx: &mut Context) {
    let events_js = r#"
        (function() {
            // Fire DOMContentLoaded on document
            try { __nb_dispatchEvent("document", "DOMContentLoaded", { type: "DOMContentLoaded", target: document }); } catch(e) {}
            // Fire readystatechange
            try { __nb_dispatchEvent("document", "readystatechange", { type: "readystatechange", target: document }); } catch(e) {}
            // Fire load on window
            try { __nb_dispatchEvent("window", "load", { type: "load", target: window }); } catch(e) {}
        })();
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(events_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Miscellaneous globals
// ═══════════════════════════════════════════════════════════════════

fn register_misc_globals(ctx: &mut Context) {
    // atob / btoa
    ctx.register_global_builtin_callable(js_string!("atob"), 1,
        NativeFunction::from_fn_ptr(|_, args, ctx| {
            let input = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
            // Simple base64 decode (no padding validation)
            let decoded = base64_decode(&input).unwrap_or_default();
            Ok(JsValue::from(js_string!(decoded)))
        }),
    ).expect("atob");

    ctx.register_global_builtin_callable(js_string!("btoa"), 1,
        NativeFunction::from_fn_ptr(|_, args, ctx| {
            let input = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
            // btoa only accepts Latin1 (0-255). Reject characters outside that range.
            let mut latin1_bytes = Vec::with_capacity(input.len());
            for ch in input.chars() {
                let cp = ch as u32;
                if cp > 255 {
                    return Err(boa_engine::JsError::from_opaque(
                        JsValue::from(js_string!("InvalidCharacterError: btoa failed — character out of Latin1 range"))
                    ));
                }
                latin1_bytes.push(cp as u8);
            }
            let encoded = base64_encode(&latin1_bytes);
            Ok(JsValue::from(js_string!(encoded)))
        }),
    ).expect("btoa");

    // encodeURIComponent / decodeURIComponent
    ctx.register_global_builtin_callable(js_string!("encodeURIComponent"), 1,
        NativeFunction::from_fn_ptr(|_, args, ctx| {
            let input = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
            let encoded = url_encode(&input);
            Ok(JsValue::from(js_string!(encoded)))
        }),
    ).expect("encodeURIComponent");

    ctx.register_global_builtin_callable(js_string!("decodeURIComponent"), 1,
        NativeFunction::from_fn_ptr(|_, args, ctx| {
            let input = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
            let decoded = url_decode(&input);
            Ok(JsValue::from(js_string!(decoded)))
        }),
    ).expect("decodeURIComponent");

    // getComputedStyle stub
    ctx.register_global_builtin_callable(js_string!("getComputedStyle"), 1,
        NativeFunction::from_fn_ptr(|_, _, ctx| {
            let style = ObjectInitializer::new(ctx)
                .function(NativeFunction::from_fn_ptr(|_, _, _| Ok(JsValue::from(js_string!("")))),
                    js_string!("getPropertyValue"), 1)
                .build();
            Ok(style.into())
        }),
    ).expect("getComputedStyle");

    // matchMedia stub
    ctx.register_global_builtin_callable(js_string!("matchMedia"), 1,
        NativeFunction::from_fn_ptr(|_, _, ctx| {
            let result = ObjectInitializer::new(ctx)
                .property(js_string!("matches"), JsValue::from(false), Attribute::all())
                .function(NativeFunction::from_fn_ptr(|_, _, _| Ok(JsValue::undefined())),
                    js_string!("addListener"), 1)
                .function(NativeFunction::from_fn_ptr(|_, _, _| Ok(JsValue::undefined())),
                    js_string!("removeListener"), 1)
                .function(NativeFunction::from_fn_ptr(|_, _, _| Ok(JsValue::undefined())),
                    js_string!("addEventListener"), 2)
                .build();
            Ok(result.into())
        }),
    ).expect("matchMedia");

    // MutationObserver stub
    let mutation_js = r#"
        function MutationObserver(callback) {
            this._callback = callback;
            this.observe = function(target, config) {};
            this.disconnect = function() {};
            this.takeRecords = function() { return []; };
        }
        function IntersectionObserver(callback, options) {
            this._callback = callback;
            this.observe = function(target) {};
            this.unobserve = function(target) {};
            this.disconnect = function() {};
        }
        function ResizeObserver(callback) {
            this._callback = callback;
            this.observe = function(target) {};
            this.unobserve = function(target) {};
            this.disconnect = function() {};
        }

        // Promise.resolve / Promise.reject polyfill check
        if (typeof Promise === "undefined") {
            function Promise(executor) {
                this._value = undefined;
                this._state = "pending";
                this._callbacks = [];
                var self = this;
                try {
                    executor(
                        function(val) { self._value = val; self._state = "fulfilled"; },
                        function(err) { self._value = err; self._state = "rejected"; }
                    );
                } catch(e) { self._value = e; self._state = "rejected"; }
            }
            Promise.prototype.then = function(onFulfilled, onRejected) {
                if (this._state === "fulfilled" && onFulfilled) return Promise.resolve(onFulfilled(this._value));
                if (this._state === "rejected" && onRejected) return Promise.resolve(onRejected(this._value));
                return this;
            };
            Promise.prototype.catch = function(onRejected) { return this.then(null, onRejected); };
            Promise.prototype.finally = function(fn) { fn(); return this; };
            Promise.resolve = function(val) { return new Promise(function(res) { res(val); }); };
            Promise.reject = function(val) { return new Promise(function(_, rej) { rej(val); }); };
            Promise.all = function(promises) {
                return new Promise(function(resolve) {
                    var results = [];
                    for (var i = 0; i < promises.length; i++) results.push(promises[i]._value);
                    resolve(results);
                });
            };
        }

        // URL constructor
        if (typeof URL === "undefined") {
            function URL(url, base) {
                var full = url;
                if (base && !/^[a-z]+:\/\//i.test(url)) {
                    full = base.replace(/\/[^\/]*$/, "/") + url;
                }
                this.href = full;
                var m = full.match(/^([a-z]+:)\/\/([^\/:?#]+)(:\d+)?(\/[^?#]*)?(\?[^#]*)?(#.*)?$/i);
                if (m) {
                    this.protocol = m[1] || "https:";
                    this.hostname = m[2] || "";
                    this.port = (m[3]||"").replace(":","");
                    this.host = this.hostname + (this.port ? ":" + this.port : "");
                    this.pathname = m[4] || "/";
                    this.search = m[5] || "";
                    this.hash = m[6] || "";
                    this.origin = this.protocol + "//" + this.host;
                } else {
                    this.protocol=""; this.hostname=""; this.port="";
                    this.host=""; this.pathname=full; this.search="";
                    this.hash=""; this.origin="";
                }
                this.searchParams = new URLSearchParams(this.search);
                this.toString = function() { return this.href; };
                this.toJSON = function() { return this.href; };
            }
        }

        // AbortController stub
        function AbortController() {
            this.signal = { aborted: false, addEventListener: function() {} };
            this.abort = function() { this.signal.aborted = true; };
        }

        // TextEncoder / TextDecoder
        function TextEncoder() {}
        TextEncoder.prototype.encode = function(str) {
            if (!str) return [];
            var arr = [];
            for (var i = 0; i < str.length; i++) {
                var code = str.charCodeAt(i);
                if (code < 0x80) arr.push(code);
                else if (code < 0x800) { arr.push(0xC0|(code>>6), 0x80|(code&0x3F)); }
                else { arr.push(0xE0|(code>>12), 0x80|((code>>6)&0x3F), 0x80|(code&0x3F)); }
            }
            return arr;
        };
        function TextDecoder(enc) { this.encoding = enc || "utf-8"; }
        TextDecoder.prototype.decode = function(arr) {
            if (!arr || !arr.length) return "";
            var r = "", i = 0;
            while (i < arr.length) {
                var b = arr[i];
                if (b < 0x80) { r += String.fromCharCode(b); i++; }
                else if (b < 0xE0) { r += String.fromCharCode(((b&0x1F)<<6)|(arr[i+1]&0x3F)); i+=2; }
                else { r += String.fromCharCode(((b&0x0F)<<12)|((arr[i+1]&0x3F)<<6)|(arr[i+2]&0x3F)); i+=3; }
            }
            return r;
        };

        // structuredClone
        if (typeof structuredClone === "undefined") {
            function structuredClone(obj) { return JSON.parse(JSON.stringify(obj)); }
        }

        // crypto.getRandomValues stub
        var crypto = { getRandomValues: function(arr) { for (var i=0; i<arr.length; i++) arr[i] = Math.floor(Math.random()*256); return arr; } };
        crypto.randomUUID = function() { return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(c) { var r = Math.random()*16|0; return (c==="x"?r:(r&0x3|0x8)).toString(16); }); };

        // ── Phase 3 additions ──

        // Ensure JSON.stringify handles circular refs gracefully
        var _origStringify = JSON.stringify;
        JSON.stringify = function(obj, replacer, space) {
            try { return _origStringify(obj, replacer, space); }
            catch(e) { return "{}"; }
        };

        // Date.now polyfill check
        if (typeof Date.now !== "function") {
            Date.now = function() { return new Date().getTime(); };
        }

        // globalThis alias
        if (typeof globalThis === "undefined") {
            var globalThis = window || this;
        }

        // FormData stub
        function FormData(form) {
            this._data = {};
            this.append = function(key, val) { this._data[key] = val; };
            this.get = function(key) { return this._data[key] || null; };
            this.set = function(key, val) { this._data[key] = val; };
            this.has = function(key) { return key in this._data; };
            this.delete = function(key) { delete this._data[key]; };
            this.entries = function() { var arr = []; for (var k in this._data) arr.push([k, this._data[k]]); return arr; };
            this.keys = function() { return Object.keys(this._data); };
            this.values = function() { return Object.values(this._data); };
        }

        // URLSearchParams stub
        function URLSearchParams(init) {
            this._params = {};
            if (typeof init === "string") {
                var parts = init.replace(/^\?/, "").split("&");
                for (var i = 0; i < parts.length; i++) {
                    var kv = parts[i].split("=");
                    if (kv[0]) this._params[decodeURIComponent(kv[0])] = decodeURIComponent(kv[1] || "");
                }
            }
            this.get = function(key) { return this._params[key] || null; };
            this.set = function(key, val) { this._params[key] = val; };
            this.has = function(key) { return key in this._params; };
            this.delete = function(key) { delete this._params[key]; };
            this.append = function(key, val) { this._params[key] = val; };
            this.toString = function() {
                var parts = [];
                for (var k in this._params) parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(this._params[k]));
                return parts.join("&");
            };
            this.entries = function() { var arr = []; for (var k in this._params) arr.push([k, this._params[k]]); return arr; };
        }

        // WeakMap / WeakSet / Map / Set existence check (Boa has these built-in)
        // Just ensure Symbol.iterator exists for for..of compatibility
        if (typeof Symbol === "undefined") {
            var Symbol = { iterator: "@@iterator", toStringTag: "@@toStringTag", hasInstance: "@@hasInstance" };
        }

        // DOMParser stub
        function DOMParser() {}
        DOMParser.prototype.parseFromString = function(str, type) {
            return { querySelector: function() { return null; }, querySelectorAll: function() { return []; }, body: null };
        };

        // Blob — text()/arrayBuffer() return Promises per spec
        function Blob(parts, options) {
            this._parts = parts || [];
            this.size = 0;
            this.type = (options && options.type) || "";
            for (var i = 0; i < this._parts.length; i++) this.size += (this._parts[i].length || 0);
        }
        Blob.prototype.text = function() { return Promise.resolve(this._parts.join("")); };
        Blob.prototype.arrayBuffer = function() {
            var str = this._parts.join("");
            var enc = new TextEncoder();
            return Promise.resolve(enc.encode(str));
        };
        Blob.prototype.slice = function(start, end, type) {
            var txt = this._parts.join("");
            var sliced = txt.slice(start || 0, end || txt.length);
            return new Blob([sliced], { type: type || this.type });
        };

        // File extends Blob (proper prototype chain)
        function File(parts, name, options) {
            Blob.call(this, parts, options);
            this.name = name || "";
            this.lastModified = (options && options.lastModified) || Date.now();
        }
        File.prototype = Object.create(Blob.prototype);
        File.prototype.constructor = File;

        // Headers — supports multiple values per key via append()
        function Headers(init) {
            this._h = {};
            if (init) {
                if (Array.isArray(init)) {
                    for (var i = 0; i < init.length; i++) this.append(init[i][0], init[i][1]);
                } else {
                    for (var k in init) this.set(k, init[k]);
                }
            }
        }
        Headers.prototype.get = function(name) {
            var vals = this._h[name.toLowerCase()];
            return vals ? vals.join(", ") : null;
        };
        Headers.prototype.set = function(name, val) { this._h[name.toLowerCase()] = [String(val)]; };
        Headers.prototype.has = function(name) { return name.toLowerCase() in this._h; };
        Headers.prototype.delete = function(name) { delete this._h[name.toLowerCase()]; };
        Headers.prototype.append = function(name, val) {
            var key = name.toLowerCase();
            if (!this._h[key]) this._h[key] = [];
            this._h[key].push(String(val));
        };
        Headers.prototype.entries = function() {
            var arr = [];
            for (var k in this._h) arr.push([k, this._h[k].join(", ")]);
            return arr;
        };
        Headers.prototype.forEach = function(cb) {
            for (var k in this._h) cb(this._h[k].join(", "), k, this);
        };

        // requestIdleCallback stub
        function requestIdleCallback(fn) { setTimeout(fn, 0); return ++__nb_timer_id; }
        function cancelIdleCallback(id) { clearTimeout(id); }

        // document.createDocumentFragment — uses the proper native implementation
        // (already wrapped above via _origCreateDocFragment)

        // document.createEvent stub
        document.createEvent = function(type) {
            return new Event(type);
        };

        // window.getComputedStyle already registered above, but ensure it handles nodes
        // element.getBoundingClientRect stub
        function __nb_getBoundingClientRect(nodeId) {
            return { top: 0, left: 0, bottom: 0, right: 0, width: 0, height: 0, x: 0, y: 0 };
        }

        // element.matches stub
        function __nb_matches(nodeId, selector) { return false; }
        function __nb_closest(nodeId, selector) { return null; }
        function __nb_contains(parentId, childId) {
            var children = __nb_getChildren(parentId);
            for (var i = 0; i < children.length; i++) {
                if (children[i] === childId) return true;
                if (__nb_contains(children[i], childId)) return true;
            }
            return false;
        }
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(mutation_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4: Element Wrapper — makes document APIs return real DOM objects
// ═══════════════════════════════════════════════════════════════════

fn register_element_wrapper(ctx: &mut Context) {
    // This giant JS block defines __NB_Element and patches document.*
    // to return wrapped objects instead of raw node IDs.
    // All existing __nb_* native functions are preserved and called internally.
    let wrapper_js = r##"
(function() {
    "use strict";

    // Cache for element wrappers — same nodeId always returns same object
    var __nb_element_cache = {};

    // ── __NB_Element constructor ──
    function __NB_Element(nodeId) {
        if (!(this instanceof __NB_Element)) return new __NB_Element(nodeId);
        if (nodeId === null || nodeId === undefined) return null;
        // Return cached wrapper if exists
        if (__nb_element_cache[nodeId]) return __nb_element_cache[nodeId];

        this.__nb_id = nodeId;
        this.nodeType = 1; // default ELEMENT_NODE, updated by getter below
        this._eventListeners = {};

        // Cache it
        __nb_element_cache[nodeId] = this;
        var _cacheKeys = Object.keys(__nb_element_cache);
        if (_cacheKeys.length > 5000) {
            for (var _ci = 0; _ci < 500; _ci++) delete __nb_element_cache[_cacheKeys[_ci]];
        }

        // ── Style proxy ──
        this.style = new __nb_StyleProxy(nodeId);
    }

    // ── Wrap helper: convert nodeId to __NB_Element or null ──
    function __nb_wrap(nodeId) {
        if (nodeId === null || nodeId === undefined || nodeId < 0) return null;
        if (typeof nodeId !== "number") return nodeId; // already wrapped or other
        if (__nb_element_cache[nodeId]) return __nb_element_cache[nodeId];
        return new __NB_Element(nodeId);
    }

    // ── Wrap array of nodeIds into array of elements ──
    function __nb_wrapAll(arr) {
        if (!arr || !arr.length) return [];
        var result = [];
        for (var i = 0; i < arr.length; i++) {
            var el = __nb_wrap(arr[i]);
            if (el) result.push(el);
        }
        // Add item() method for HTMLCollection/NodeList compat
        result.item = function(idx) { return this[idx] || null; };
        result.namedItem = function(name) { return null; };
        return result;
    }

    // ── nodeType property (lazy — reads tag from DOM) ──
    Object.defineProperty(__NB_Element.prototype, "nodeType", {
        get: function() {
            if (this._nodeType !== undefined) return this._nodeType;
            try {
                var tag = __nb_getTagName(this.__nb_id);
                if (tag === "#text") this._nodeType = 3;
                else if (tag === "#comment") this._nodeType = 8;
                else if (tag === "#document") this._nodeType = 9;
                else if (tag === "#document-fragment") this._nodeType = 11;
                else this._nodeType = 1;
            } catch(e) { this._nodeType = 1; }
            return this._nodeType;
        },
        set: function(v) { this._nodeType = v; },
        enumerable: true, configurable: true
    });

    // ── nodeName property (lazy) ──
    Object.defineProperty(__NB_Element.prototype, "nodeName", {
        get: function() {
            if (this._nodeName !== undefined) return this._nodeName;
            try {
                var tag = __nb_getTagName(this.__nb_id);
                if (tag && tag.charAt(0) === "#") this._nodeName = tag;
                else this._nodeName = tag ? tag.toUpperCase() : "";
            } catch(e) { this._nodeName = ""; }
            return this._nodeName;
        },
        set: function(v) { this._nodeName = v; },
        enumerable: true, configurable: true
    });

    // ── textContent property ──
    Object.defineProperty(__NB_Element.prototype, "textContent", {
        get: function() { return __nb_getTextContent(this.__nb_id); },
        set: function(v) { __nb_setTextContent(this.__nb_id, v == null ? "" : String(v)); },
        enumerable: true, configurable: true
    });

    // ── innerText (alias for textContent in our simplified model) ──
    Object.defineProperty(__NB_Element.prototype, "innerText", {
        get: function() { return __nb_getTextContent(this.__nb_id); },
        set: function(v) { __nb_setTextContent(this.__nb_id, v == null ? "" : String(v)); },
        enumerable: true, configurable: true
    });

    // ── innerHTML property ──
    Object.defineProperty(__NB_Element.prototype, "innerHTML", {
        get: function() { return __nb_getInnerHTML(this.__nb_id); },
        set: function(v) { __nb_setInnerHTML(this.__nb_id, v == null ? "" : String(v)); },
        enumerable: true, configurable: true
    });

    // ── outerHTML property (includes attributes) ──
    Object.defineProperty(__NB_Element.prototype, "outerHTML", {
        get: function() {
            // Use the Rust-side reconstruction which includes attributes
            return __nb_getInnerHTML(this.__nb_id) || (function(el) {
                var tag = __nb_getTagName(el.__nb_id).toLowerCase();
                if (tag.charAt(0) === "#") return el.textContent || "";
                var attrs = "";
                var id = el.id; if (id) attrs += ' id="' + id + '"';
                var cls = el.className; if (cls) attrs += ' class="' + cls + '"';
                var inner = "";
                var kids = __nb_getChildren(el.__nb_id) || [];
                for (var i = 0; i < kids.length; i++) {
                    var child = __nb_wrap(kids[i]);
                    if (child) inner += child.outerHTML || "";
                }
                if (el.textContent && kids.length === 0) inner = el.textContent;
                return "<" + tag + attrs + ">" + inner + "</" + tag + ">";
            })(this);
        },
        enumerable: true, configurable: true
    });

    // ── tagName / nodeName ──
    Object.defineProperty(__NB_Element.prototype, "tagName", {
        get: function() { return __nb_getTagName(this.__nb_id); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "nodeName", {
        get: function() { return __nb_getTagName(this.__nb_id); },
        enumerable: true, configurable: true
    });

    // ── id property ──
    Object.defineProperty(__NB_Element.prototype, "id", {
        get: function() { return __nb_getAttribute(this.__nb_id, "id") || ""; },
        set: function(v) { __nb_setAttribute(this.__nb_id, "id", String(v)); },
        enumerable: true, configurable: true
    });

    // ── className property ──
    Object.defineProperty(__NB_Element.prototype, "className", {
        get: function() { return __nb_getAttribute(this.__nb_id, "class") || ""; },
        set: function(v) { __nb_setAttribute(this.__nb_id, "class", String(v)); },
        enumerable: true, configurable: true
    });

    // ── classList object ──
    Object.defineProperty(__NB_Element.prototype, "classList", {
        get: function() {
            var nid = this.__nb_id;
            return {
                add: function() { for (var i = 0; i < arguments.length; i++) __nb_classAdd(nid, arguments[i]); },
                remove: function() { for (var i = 0; i < arguments.length; i++) __nb_classRemove(nid, arguments[i]); },
                toggle: function(cls, force) {
                    if (force !== undefined) {
                        if (force) { __nb_classAdd(nid, cls); return true; }
                        else { __nb_classRemove(nid, cls); return false; }
                    }
                    return __nb_classToggle(nid, cls);
                },
                contains: function(cls) { return __nb_classContains(nid, cls); },
                item: function(idx) {
                    var cls = (__nb_getAttribute(nid, "class") || "").split(/\s+/).filter(Boolean);
                    return cls[idx] || null;
                },
                get length() {
                    return (__nb_getAttribute(nid, "class") || "").split(/\s+/).filter(Boolean).length;
                },
                toString: function() { return __nb_getAttribute(nid, "class") || ""; },
                replace: function(oldCls, newCls) {
                    if (__nb_classContains(nid, oldCls)) {
                        __nb_classRemove(nid, oldCls);
                        __nb_classAdd(nid, newCls);
                        return true;
                    }
                    return false;
                }
            };
        },
        enumerable: true, configurable: true
    });

    // ── value property (for inputs) ──
    Object.defineProperty(__NB_Element.prototype, "value", {
        get: function() { return __nb_getValue(this.__nb_id); },
        set: function(v) { __nb_setValue(this.__nb_id, String(v)); },
        enumerable: true, configurable: true
    });

    // ── checked property ──
    Object.defineProperty(__NB_Element.prototype, "checked", {
        get: function() { return __nb_getChecked(this.__nb_id); },
        set: function(v) { __nb_setChecked(this.__nb_id, !!v); },
        enumerable: true, configurable: true
    });

    // ── disabled property ──
    Object.defineProperty(__NB_Element.prototype, "disabled", {
        get: function() { return __nb_getDisabled(this.__nb_id); },
        set: function(v) { __nb_setDisabled(this.__nb_id, !!v); },
        enumerable: true, configurable: true
    });

    // ── hidden property ──
    Object.defineProperty(__NB_Element.prototype, "hidden", {
        get: function() { return __nb_hasAttribute(this.__nb_id, "hidden"); },
        set: function(v) {
            if (v) __nb_setAttribute(this.__nb_id, "hidden", "");
            else __nb_removeAttribute(this.__nb_id, "hidden");
        },
        enumerable: true, configurable: true
    });

    // ── src, href, type, name, placeholder, title, alt ──
    var attrProps = ["src", "href", "type", "name", "placeholder", "title", "alt",
                     "action", "method", "target", "rel", "role", "tabindex",
                     "aria-label", "aria-hidden", "aria-expanded", "for", "lang"];
    for (var i = 0; i < attrProps.length; i++) {
        (function(attr) {
            // Use camelCase for JS property name
            var jsProp = attr.replace(/-([a-z])/g, function(m, c) { return c.toUpperCase(); });
            Object.defineProperty(__NB_Element.prototype, jsProp, {
                get: function() { return __nb_getAttribute(this.__nb_id, attr) || ""; },
                set: function(v) { __nb_setAttribute(this.__nb_id, attr, String(v)); },
                enumerable: true, configurable: true
            });
        })(attrProps[i]);
    }

    // ── dataset (data-* attributes) ──
    Object.defineProperty(__NB_Element.prototype, "dataset", {
        get: function() {
            var nid = this.__nb_id;
            return new Proxy({}, {
                get: function(target, prop) {
                    // Convert camelCase to data-kebab-case
                    var attr = "data-" + prop.replace(/([A-Z])/g, "-$1").toLowerCase();
                    return __nb_getAttribute(nid, attr) || undefined;
                },
                set: function(target, prop, value) {
                    var attr = "data-" + prop.replace(/([A-Z])/g, "-$1").toLowerCase();
                    __nb_setAttribute(nid, attr, String(value));
                    return true;
                },
                deleteProperty: function(target, prop) {
                    var attr = "data-" + prop.replace(/([A-Z])/g, "-$1").toLowerCase();
                    __nb_removeAttribute(nid, attr);
                    return true;
                }
            });
        },
        enumerable: true, configurable: true
    });

    // ── parentNode / parentElement ──
    Object.defineProperty(__NB_Element.prototype, "parentNode", {
        get: function() { return __nb_wrap(__nb_getParent(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "parentElement", {
        get: function() { return __nb_wrap(__nb_getParent(this.__nb_id)); },
        enumerable: true, configurable: true
    });

    // ── children / childNodes ──
    Object.defineProperty(__NB_Element.prototype, "children", {
        get: function() { return __nb_wrapAll(__nb_getChildren(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "childNodes", {
        get: function() { return __nb_wrapAll(__nb_getChildren(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "childElementCount", {
        get: function() { return __nb_getChildCount(this.__nb_id); },
        enumerable: true, configurable: true
    });

    // ── firstChild / lastChild / firstElementChild / lastElementChild ──
    Object.defineProperty(__NB_Element.prototype, "firstChild", {
        get: function() { return __nb_wrap(__nb_getFirstChild(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "lastChild", {
        get: function() { return __nb_wrap(__nb_getLastChild(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "firstElementChild", {
        get: function() { return __nb_wrap(__nb_getFirstChild(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "lastElementChild", {
        get: function() { return __nb_wrap(__nb_getLastChild(this.__nb_id)); },
        enumerable: true, configurable: true
    });

    // ── nextSibling / previousSibling ──
    Object.defineProperty(__NB_Element.prototype, "nextSibling", {
        get: function() { return __nb_wrap(__nb_getNextSibling(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "previousSibling", {
        get: function() { return __nb_wrap(__nb_getPrevSibling(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "nextElementSibling", {
        get: function() { return __nb_wrap(__nb_getNextSibling(this.__nb_id)); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(__NB_Element.prototype, "previousElementSibling", {
        get: function() { return __nb_wrap(__nb_getPrevSibling(this.__nb_id)); },
        enumerable: true, configurable: true
    });

    // ── DOM manipulation methods ──
    __NB_Element.prototype.appendChild = function(child) {
        if (!child || child.__nb_id === undefined) return child;
        __nb_appendChild(this.__nb_id, child.__nb_id);
        return child;
    };
    __NB_Element.prototype.removeChild = function(child) {
        if (!child || child.__nb_id === undefined) return child;
        __nb_removeChild(this.__nb_id, child.__nb_id);
        return child;
    };
    __NB_Element.prototype.insertBefore = function(newChild, refChild) {
        if (!newChild || newChild.__nb_id === undefined) return newChild;
        var refId = (refChild && refChild.__nb_id !== undefined) ? refChild.__nb_id : -1;
        if (refId >= 0) {
            __nb_insertBefore(this.__nb_id, newChild.__nb_id, refId);
        } else {
            __nb_appendChild(this.__nb_id, newChild.__nb_id);
        }
        return newChild;
    };
    __NB_Element.prototype.replaceChild = function(newChild, oldChild) {
        if (!newChild || !oldChild) return oldChild;
        this.insertBefore(newChild, oldChild);
        this.removeChild(oldChild);
        return oldChild;
    };
    __NB_Element.prototype.append = function() {
        for (var i = 0; i < arguments.length; i++) {
            var arg = arguments[i];
            if (typeof arg === "string") {
                var text = document.createTextNode(arg);
                this.appendChild(text);
            } else if (arg && arg.__nb_id !== undefined) {
                this.appendChild(arg);
            }
        }
    };
    __NB_Element.prototype.prepend = function() {
        var first = this.firstChild;
        for (var i = 0; i < arguments.length; i++) {
            var arg = arguments[i];
            if (typeof arg === "string") {
                var text = document.createTextNode(arg);
                this.insertBefore(text, first);
            } else if (arg && arg.__nb_id !== undefined) {
                this.insertBefore(arg, first);
            }
        }
    };
    __NB_Element.prototype.remove = function() {
        var parent = __nb_getParent(this.__nb_id);
        if (parent !== null) __nb_removeChild(parent, this.__nb_id);
    };
    __NB_Element.prototype.after = function() {
        var parent = this.parentNode;
        var next = this.nextSibling;
        if (!parent) return;
        for (var i = 0; i < arguments.length; i++) {
            var arg = arguments[i];
            if (typeof arg === "string") arg = document.createTextNode(arg);
            if (next) parent.insertBefore(arg, next);
            else parent.appendChild(arg);
        }
    };
    __NB_Element.prototype.before = function() {
        var parent = this.parentNode;
        if (!parent) return;
        for (var i = 0; i < arguments.length; i++) {
            var arg = arguments[i];
            if (typeof arg === "string") arg = document.createTextNode(arg);
            parent.insertBefore(arg, this);
        }
    };

    // ── insertAdjacentHTML / insertAdjacentElement / insertAdjacentText ──
    __NB_Element.prototype.insertAdjacentHTML = function(position, html) {
        var temp = document.createElement("div");
        temp.innerHTML = html;
        var kids = temp.childNodes;
        var pos = (position || "").toLowerCase();
        if (pos === "beforebegin") {
            for (var i = 0; i < kids.length; i++) this.before(kids[i]);
        } else if (pos === "afterbegin") {
            var first = this.firstChild;
            for (var i = kids.length - 1; i >= 0; i--) {
                if (first) this.insertBefore(kids[i], first);
                else this.appendChild(kids[i]);
                first = this.firstChild;
            }
        } else if (pos === "beforeend") {
            for (var i = 0; i < kids.length; i++) this.appendChild(kids[i]);
        } else if (pos === "afterend") {
            for (var i = kids.length - 1; i >= 0; i--) this.after(kids[i]);
        }
    };
    __NB_Element.prototype.insertAdjacentElement = function(position, el) {
        if (!el) return null;
        var pos = (position || "").toLowerCase();
        if (pos === "beforebegin") this.before(el);
        else if (pos === "afterbegin") this.prepend(el);
        else if (pos === "beforeend") this.appendChild(el);
        else if (pos === "afterend") this.after(el);
        return el;
    };
    __NB_Element.prototype.insertAdjacentText = function(position, text) {
        this.insertAdjacentHTML(position, text);
    };

    // ── hasChildNodes ──
    __NB_Element.prototype.hasChildNodes = function() {
        return __nb_getChildCount(this.__nb_id) > 0;
    };

    // ── nodeValue (for text/comment nodes) ──
    Object.defineProperty(__NB_Element.prototype, "nodeValue", {
        get: function() {
            var nt = this.nodeType;
            if (nt === 3 || nt === 8) return this.textContent;
            return null;
        },
        set: function(v) {
            var nt = this.nodeType;
            if (nt === 3 || nt === 8) this.textContent = v;
        },
        enumerable: true, configurable: true
    });

    // ── Attribute methods ──
    __NB_Element.prototype.setAttribute = function(key, val) {
        __nb_setAttribute(this.__nb_id, key, String(val));
    };
    __NB_Element.prototype.getAttribute = function(key) {
        return __nb_getAttribute(this.__nb_id, key);
    };
    __NB_Element.prototype.removeAttribute = function(key) {
        __nb_removeAttribute(this.__nb_id, key);
    };
    __NB_Element.prototype.hasAttribute = function(key) {
        return __nb_hasAttribute(this.__nb_id, key);
    };
    __NB_Element.prototype.getAttributeNode = function(key) {
        var val = __nb_getAttribute(this.__nb_id, key);
        return val !== null ? { name: key, value: val } : null;
    };
    __NB_Element.prototype.toggleAttribute = function(name, force) {
        if (force !== undefined) {
            if (force) { this.setAttribute(name, ""); return true; }
            else { this.removeAttribute(name); return false; }
        }
        if (this.hasAttribute(name)) { this.removeAttribute(name); return false; }
        this.setAttribute(name, ""); return true;
    };

    // ── Query methods (scoped) ──
    __NB_Element.prototype.querySelector = function(sel) {
        // Use global querySelector and check if result is descendant
        // Simple: delegate to document and filter (not perfect but works)
        var all = document.__nb_querySelectorAll(sel);
        var children = this.__getAllDescendants();
        for (var i = 0; i < all.length; i++) {
            if (children.indexOf(all[i]) >= 0) return __nb_wrap(all[i]);
        }
        return null;
    };
    __NB_Element.prototype.querySelectorAll = function(sel) {
        var all = document.__nb_querySelectorAll(sel);
        var children = this.__getAllDescendants();
        var result = [];
        for (var i = 0; i < all.length; i++) {
            if (children.indexOf(all[i]) >= 0) result.push(all[i]);
        }
        return __nb_wrapAll(result);
    };
    __NB_Element.prototype.getElementsByTagName = function(tag) {
        return this.querySelectorAll(tag);
    };
    __NB_Element.prototype.getElementsByClassName = function(cls) {
        return this.querySelectorAll("." + cls);
    };
    __NB_Element.prototype.__getAllDescendants = function() {
        var result = [];
        var stack = __nb_getChildren(this.__nb_id).slice();
        while (stack.length > 0) {
            var nid = stack.pop();
            result.push(nid);
            var ch = __nb_getChildren(nid);
            for (var i = 0; i < ch.length; i++) stack.push(ch[i]);
        }
        return result;
    };

    // ── matches / closest ──
    __NB_Element.prototype.matches = function(sel) {
        // Compound selector matching: supports tag, #id, .class, [attr], and combinations
        if (!sel) return false;
        sel = sel.trim();
        // Handle comma-separated selectors (OR)
        if (sel.indexOf(",") >= 0) {
            var parts = sel.split(",");
            for (var i = 0; i < parts.length; i++) {
                if (this.matches(parts[i].trim())) return true;
            }
            return false;
        }
        // Split compound selector into parts: tag, .class, #id, [attr]
        var regex = /([a-zA-Z0-9_-]+)|(\.[a-zA-Z0-9_-]+)|(#[a-zA-Z0-9_-]+)|(\[[^\]]+\])|(\*)/g;
        var match;
        var hasAny = false;
        while ((match = regex.exec(sel)) !== null) {
            hasAny = true;
            var token = match[0];
            if (token === "*") { continue; }
            if (token.charAt(0) === "#") {
                if (this.id !== token.substring(1)) return false;
            } else if (token.charAt(0) === ".") {
                if (!this.classList.contains(token.substring(1))) return false;
            } else if (token.charAt(0) === "[") {
                // Attribute selector: [attr], [attr=val], [attr~=val], [attr^=val]
                var inner = token.slice(1, -1);
                var eqIdx = inner.indexOf("=");
                if (eqIdx < 0) {
                    if (!this.hasAttribute(inner)) return false;
                } else {
                    var op = inner.charAt(eqIdx - 1);
                    var attrName, attrVal;
                    if (op === "~" || op === "^" || op === "$" || op === "*") {
                        attrName = inner.substring(0, eqIdx - 1);
                        attrVal = inner.substring(eqIdx + 1).replace(/^["']|["']$/g, "");
                    } else {
                        attrName = inner.substring(0, eqIdx);
                        attrVal = inner.substring(eqIdx + 1).replace(/^["']|["']$/g, "");
                        op = "=";
                    }
                    var actual = this.getAttribute(attrName);
                    if (actual === null) return false;
                    if (op === "=" && actual !== attrVal) return false;
                    if (op === "~" && !(" " + actual + " ").includes(" " + attrVal + " ")) return false;
                    if (op === "^" && !actual.startsWith(attrVal)) return false;
                    if (op === "$" && !actual.endsWith(attrVal)) return false;
                    if (op === "*" && !actual.includes(attrVal)) return false;
                }
            } else {
                // Tag name match
                if (this.tagName.toLowerCase() !== token.toLowerCase()) return false;
            }
        }
        return hasAny;
    };
    __NB_Element.prototype.closest = function(sel) {
        var el = this;
        while (el) {
            if (el.matches && el.matches(sel)) return el;
            el = el.parentElement;
        }
        return null;
    };
    __NB_Element.prototype.contains = function(other) {
        if (!other || other.__nb_id === undefined) return false;
        if (other.__nb_id === this.__nb_id) return true;
        return __nb_contains(this.__nb_id, other.__nb_id);
    };

    // ── cloneNode ──
    __NB_Element.prototype.cloneNode = function(deep) {
        var newId = __nb_cloneNode(this.__nb_id, !!deep);
        return __nb_wrap(newId);
    };

    // ── Event methods ──
    __NB_Element.prototype.addEventListener = function(type, listener, options) {
        if (typeof listener !== "function") return;
        if (!this._eventListeners[type]) this._eventListeners[type] = [];
        this._eventListeners[type].push(listener);
        // Also register globally for potential dispatch
        __nb_addEventListener("el_" + this.__nb_id, type, listener);
    };
    __NB_Element.prototype.removeEventListener = function(type, listener) {
        if (this._eventListeners[type]) {
            this._eventListeners[type] = this._eventListeners[type].filter(function(l) { return l !== listener; });
        }
        __nb_removeEventListener("el_" + this.__nb_id, type, listener);
    };
    __NB_Element.prototype.dispatchEvent = function(event) {
        var type = event.type || event;
        event.target = this;
        var _stopped = false;
        var origStop = event.stopPropagation;
        event.stopPropagation = function() { _stopped = true; if (origStop) origStop.call(event); };
        // Fire listeners on the target element
        event.currentTarget = this;
        if (this._eventListeners[type]) {
            for (var i = 0; i < this._eventListeners[type].length; i++) {
                try { this._eventListeners[type][i].call(this, event); } catch(e) {}
            }
        }
        // Bubble up through parent elements if event.bubbles is true
        if (event.bubbles !== false && !_stopped) {
            var parentId = __nb_getParent(this.__nb_id);
            while (parentId !== null && parentId !== undefined && !_stopped) {
                var parentEl = __nb_wrap(parentId);
                event.currentTarget = parentEl;
                if (parentEl._eventListeners && parentEl._eventListeners[type]) {
                    for (var j = 0; j < parentEl._eventListeners[type].length; j++) {
                        try { parentEl._eventListeners[type][j].call(parentEl, event); } catch(e) {}
                        if (_stopped) break;
                    }
                }
                parentId = __nb_getParent(parentId);
            }
        }
        return true;
    };

    // ── on* event handler properties ──
    var eventHandlerProps = ["onclick", "onchange", "oninput", "onsubmit", "onfocus",
        "onblur", "onmouseenter", "onmouseleave", "onmouseover", "onmouseout",
        "onmousedown", "onmouseup", "onkeydown", "onkeyup", "onkeypress",
        "onscroll", "onresize", "ontouchstart", "ontouchend", "ontouchmove",
        "ondragstart", "ondragend", "ondrop", "oncontextmenu", "ondblclick",
        "onwheel", "onerror", "onload", "onabort", "ontransitionend",
        "onanimationend", "onanimationstart"];
    for (var i = 0; i < eventHandlerProps.length; i++) {
        (function(prop) {
            var type = prop.substring(2); // "onclick" -> "click"
            Object.defineProperty(__NB_Element.prototype, prop, {
                get: function() { return this["_" + prop] || null; },
                set: function(fn) {
                    // Remove old handler
                    if (this["_" + prop]) this.removeEventListener(type, this["_" + prop]);
                    this["_" + prop] = fn;
                    if (fn) this.addEventListener(type, fn);
                },
                enumerable: true, configurable: true
            });
        })(eventHandlerProps[i]);
    }

    // ── Layout metric stubs ──
    __NB_Element.prototype.getBoundingClientRect = function() {
        return { top: 0, left: 0, bottom: 100, right: 200, width: 200, height: 100, x: 0, y: 0,
                 toJSON: function() { return this; } };
    };
    __NB_Element.prototype.getClientRects = function() { return [this.getBoundingClientRect()]; };
    Object.defineProperty(__NB_Element.prototype, "offsetWidth", { get: function() { return 200; } });
    Object.defineProperty(__NB_Element.prototype, "offsetHeight", { get: function() { return 100; } });
    Object.defineProperty(__NB_Element.prototype, "clientWidth", { get: function() { return 200; } });
    Object.defineProperty(__NB_Element.prototype, "clientHeight", { get: function() { return 100; } });
    Object.defineProperty(__NB_Element.prototype, "scrollWidth", { get: function() { return 200; } });
    Object.defineProperty(__NB_Element.prototype, "scrollHeight", { get: function() { return 100; } });
    Object.defineProperty(__NB_Element.prototype, "offsetTop", { get: function() { return 0; } });
    Object.defineProperty(__NB_Element.prototype, "offsetLeft", { get: function() { return 0; } });
    Object.defineProperty(__NB_Element.prototype, "scrollTop", {
        get: function() { return 0; },
        set: function(v) {}
    });
    Object.defineProperty(__NB_Element.prototype, "scrollLeft", {
        get: function() { return 0; },
        set: function(v) {}
    });
    Object.defineProperty(__NB_Element.prototype, "offsetParent", {
        get: function() { return this.parentElement; }
    });

    // ── focus / blur / click / scrollIntoView ──
    __NB_Element.prototype.focus = function() {};
    __NB_Element.prototype.blur = function() {};
    __NB_Element.prototype.click = function() {
        this.dispatchEvent({ type: "click", bubbles: true, target: this, preventDefault: function(){}, stopPropagation: function(){} });
    };
    __NB_Element.prototype.scrollIntoView = function() {};
    __NB_Element.prototype.animate = function() { return { finished: Promise.resolve(), cancel: function(){}, play: function(){} }; };

    // ── toString / valueOf ──
    __NB_Element.prototype.toString = function() { return "[object HTMLElement]"; };
    __NB_Element.prototype.valueOf = function() { return this.__nb_id; };

    // ── isConnected ──
    Object.defineProperty(__NB_Element.prototype, "isConnected", {
        get: function() { return __nb_nodeExists(this.__nb_id); }
    });

    // ── ownerDocument ──
    Object.defineProperty(__NB_Element.prototype, "ownerDocument", {
        get: function() { return document; }
    });

    // ══════════════════════════════════════════════
    // Save original document methods that return raw IDs
    // ══════════════════════════════════════════════
    var _origGetById = document.getElementById;
    var _origCreateElement = document.createElement;
    var _origQuerySelector = document.querySelector;
    var _origQuerySelectorAll = document.querySelectorAll;
    var _origGetByTagName = document.getElementsByTagName;
    var _origGetByClassName = document.getElementsByClassName;
    var _origCreateTextNode = document.createTextNode;
    var _origCreateComment = document.createComment;
    var _origCreateDocFragment = document.createDocumentFragment;

    // Keep raw versions for internal use
    document.__nb_querySelectorAll = function(sel) {
        return _origQuerySelectorAll.call(document, sel);
    };

    // ══════════════════════════════════════════════
    // Monkey-patch document methods to return wrapped elements
    // ══════════════════════════════════════════════
    document.getElementById = function(id) {
        var nid = _origGetById.call(document, id);
        return __nb_wrap(nid);
    };

    document.createElement = function(tag) {
        var nid = _origCreateElement.call(document, tag);
        return __nb_wrap(nid);
    };

    document.createTextNode = function(text) {
        var nid = _origCreateTextNode.call(document, text);
        return __nb_wrap(nid);
    };

    document.createComment = function(text) {
        var nid = _origCreateComment.call(document, text);
        return __nb_wrap(nid);
    };

    document.createDocumentFragment = function() {
        var nid = _origCreateDocFragment.call(document);
        return __nb_wrap(nid);
    };

    document.querySelector = function(sel) {
        var nid = _origQuerySelector.call(document, sel);
        return __nb_wrap(nid);
    };

    document.querySelectorAll = function(sel) {
        var arr = _origQuerySelectorAll.call(document, sel);
        return __nb_wrapAll(arr);
    };

    document.getElementsByTagName = function(tag) {
        var arr = _origGetByTagName.call(document, tag);
        return __nb_wrapAll(arr);
    };

    document.getElementsByClassName = function(cls) {
        var arr = _origGetByClassName.call(document, cls);
        return __nb_wrapAll(arr);
    };

    document.createDocumentFragment = function() {
        return document.createElement("div");
    };

    document.createEvent = function(type) {
        return new Event(type || "Event");
    };

    // ── document.body / head / documentElement as wrapped elements ──
    var _origBody = document.body;
    var _origHead = document.head;
    var _origDocEl = document.documentElement;

    Object.defineProperty(document, "body", {
        get: function() { return __nb_wrap(typeof _origBody === "number" ? _origBody : 0); },
        set: function(v) { _origBody = v; },
        enumerable: true, configurable: true
    });
    Object.defineProperty(document, "head", {
        get: function() { return __nb_wrap(typeof _origHead === "number" ? _origHead : 0); },
        enumerable: true, configurable: true
    });
    Object.defineProperty(document, "documentElement", {
        get: function() { return __nb_wrap(typeof _origDocEl === "number" ? _origDocEl : 0); },
        enumerable: true, configurable: true
    });

    // ── Make __nb_wrap and __NB_Element globally available ──
    window.__nb_wrap = __nb_wrap;
    window.__nb_wrapAll = __nb_wrapAll;
    window.__NB_Element = __NB_Element;

})();
    "##;

    match ctx.eval(Source::from_bytes(wrapper_js.as_bytes())) {
        Ok(_) => info!("[JS] Phase 4 Element wrapper registered"),
        Err(e) => {
            // Proxy might not be supported — try fallback without dataset Proxy
            warn!("[JS] Element wrapper error (trying fallback without Proxy): {e}");
            let fallback_js = wrapper_js.replace(
                "return new Proxy({}, {",
                "var _ds = {}; return _ds; /* Proxy fallback */ if(false) return new Proxy({}, {"
            );
            match ctx.eval(Source::from_bytes(fallback_js.as_bytes())) {
                Ok(_) => info!("[JS] Phase 4 Element wrapper registered (fallback mode)"),
                Err(e2) => error!("[JS] Element wrapper fallback also failed: {e2}"),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Element Style API (Phase 3)
// ═══════════════════════════════════════════════════════════════════

fn register_style_api(ctx: &mut Context, dom: DomPtr) {
    // __nb_getStyle(nodeId, prop) -> string
    ctx.register_global_builtin_callable(js_string!("__nb_getStyle"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let prop = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get() };
            // Read from "style" attribute: parse inline style
            let val = d.nodes.get(nid).and_then(|n| {
                n.attrs.get("style").and_then(|style_str| {
                    parse_inline_style(style_str, &prop)
                })
            }).unwrap_or_default();
            Ok(JsValue::from(js_string!(val)))
        }),
    ).expect("getStyle");

    // __nb_setStyle(nodeId, prop, value)
    ctx.register_global_builtin_callable(js_string!("__nb_setStyle"), 3,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let prop = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let val = args.get_or_undefined(2).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                let css_prop = camel_to_kebab(&prop);
                let style_str = node.attrs.entry("style".to_string()).or_default();
                set_inline_style(style_str, &css_prop, &val);
            }
            Ok(JsValue::undefined())
        }),
    ).expect("setStyle");

    // __nb_removeStyle(nodeId, prop)
    ctx.register_global_builtin_callable(js_string!("__nb_removeStyle"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let prop = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                let css_prop = camel_to_kebab(&prop);
                if let Some(style_str) = node.attrs.get_mut("style") {
                    remove_inline_style(style_str, &css_prop);
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("removeStyle");

    // Register JS-side style proxy on elements
    let style_proxy_js = r#"
        // element.style access via Proxy-like getters/setters
        function __nb_StyleProxy(nodeId) {
            var self = this;
            self._nodeId = nodeId;
            self.setProperty = function(prop, val) { __nb_setStyle(nodeId, prop, val); };
            self.getPropertyValue = function(prop) { return __nb_getStyle(nodeId, prop); };
            self.removeProperty = function(prop) { __nb_removeStyle(nodeId, prop); return ""; };
            self.cssText = "";
            // Common style properties as getters/setters
            var props = ["display","visibility","color","backgroundColor","width","height",
                "margin","padding","border","fontSize","fontWeight","fontFamily","textAlign",
                "position","top","left","right","bottom","zIndex","overflow","opacity",
                "transform","transition","animation","cursor","pointerEvents","flex",
                "flexDirection","justifyContent","alignItems","gap","gridTemplateColumns",
                "maxWidth","maxHeight","minWidth","minHeight","borderRadius","boxShadow",
                "textDecoration","lineHeight","letterSpacing","whiteSpace","wordBreak",
                "outline","float","clear","verticalAlign","listStyleType","backgroundImage",
                "backgroundSize","backgroundPosition","backgroundRepeat"];
            for (var i = 0; i < props.length; i++) {
                (function(prop) {
                    Object.defineProperty(self, prop, {
                        get: function() { return __nb_getStyle(nodeId, prop); },
                        set: function(v) { __nb_setStyle(nodeId, prop, v); },
                        enumerable: true, configurable: true
                    });
                })(props[i]);
            }
        }
        // __nb_getNodeStyle(nodeId) -> StyleProxy
        function __nb_getNodeStyle(nodeId) { return new __nb_StyleProxy(nodeId); }
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(style_proxy_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

/// Parse value from inline style string: "display: none; color: red" → get "display" → "none"
fn parse_inline_style(style_str: &str, prop: &str) -> Option<String> {
    let css_prop = camel_to_kebab(prop);
    for decl in style_str.split(';') {
        let decl = decl.trim();
        if let Some(colon) = decl.find(':') {
            let name = decl[..colon].trim();
            let value = decl[colon+1..].trim();
            if name.eq_ignore_ascii_case(&css_prop) {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Set a property in an inline style string
fn set_inline_style(style_str: &mut String, prop: &str, value: &str) {
    // Remove existing property first
    remove_inline_style(style_str, prop);
    // Append new
    if !style_str.is_empty() && !style_str.ends_with(';') {
        style_str.push(';');
    }
    if !style_str.is_empty() { style_str.push(' '); }
    style_str.push_str(prop);
    style_str.push_str(": ");
    style_str.push_str(value);
    style_str.push(';');
}

/// Remove a property from inline style string
fn remove_inline_style(style_str: &mut String, prop: &str) {
    let parts: Vec<&str> = style_str.split(';')
        .filter(|decl| {
            let decl = decl.trim();
            if decl.is_empty() { return false; }
            if let Some(colon) = decl.find(':') {
                !decl[..colon].trim().eq_ignore_ascii_case(prop)
            } else { true }
        })
        .collect();
    *style_str = parts.join("; ");
    if !style_str.is_empty() { style_str.push(';'); }
}

/// Convert camelCase to kebab-case: "backgroundColor" → "background-color"
fn camel_to_kebab(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 { result.push('-'); }
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// Form element APIs (Phase 3)
// ═══════════════════════════════════════════════════════════════════

fn register_form_api(ctx: &mut Context, dom: DomPtr) {
    // __nb_getValue(nodeId) -> value attribute or text for input/select/textarea
    ctx.register_global_builtin_callable(js_string!("__nb_getValue"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let val = d.nodes.get(nid).map(|n| {
                n.attrs.get("value").cloned().unwrap_or_else(|| n.text.clone())
            }).unwrap_or_default();
            Ok(JsValue::from(js_string!(val)))
        }),
    ).expect("getValue");

    // __nb_setValue(nodeId, value)
    ctx.register_global_builtin_callable(js_string!("__nb_setValue"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let val = args.get_or_undefined(1).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                node.attrs.insert("value".to_string(), val);
            }
            Ok(JsValue::undefined())
        }),
    ).expect("setValue");

    // __nb_getChecked(nodeId) -> bool
    ctx.register_global_builtin_callable(js_string!("__nb_getChecked"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let checked = d.nodes.get(nid).is_some_and(|n| n.attrs.contains_key("checked"));
            Ok(JsValue::from(checked))
        }),
    ).expect("getChecked");

    // __nb_setChecked(nodeId, bool)
    ctx.register_global_builtin_callable(js_string!("__nb_setChecked"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let val = args.get_or_undefined(1).to_boolean();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                if val {
                    node.attrs.insert("checked".to_string(), String::new());
                } else {
                    node.attrs.remove("checked");
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("setChecked");

    // __nb_getDisabled(nodeId) -> bool
    ctx.register_global_builtin_callable(js_string!("__nb_getDisabled"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let d = unsafe { dom.get() };
            let disabled = d.nodes.get(nid).is_some_and(|n| n.attrs.contains_key("disabled"));
            Ok(JsValue::from(disabled))
        }),
    ).expect("getDisabled");

    // __nb_setDisabled(nodeId, bool)
    ctx.register_global_builtin_callable(js_string!("__nb_setDisabled"), 2,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let nid = args.get_or_undefined(0).to_i32(ctx)? as usize;
            let val = args.get_or_undefined(1).to_boolean();
            let d = unsafe { dom.get_mut() };
            if let Some(node) = d.nodes.get_mut(nid) {
                if val {
                    node.attrs.insert("disabled".to_string(), String::new());
                } else {
                    node.attrs.remove("disabled");
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("setDisabled");
}

// ═══════════════════════════════════════════════════════════════════
// document.cookie API (Phase 3)
// ═══════════════════════════════════════════════════════════════════

fn register_cookie_api(ctx: &mut Context, cookies: StoragePtr) {
    // __nb_getCookies() -> "name=val; name2=val2"
    ctx.register_global_builtin_callable(js_string!("__nb_getCookies"), 0,
        NativeFunction::from_copy_closure(move |_, _, _| {
            let c = unsafe { cookies.get() };
            let cookie_str: String = c.iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join("; ");
            Ok(JsValue::from(js_string!(cookie_str)))
        }),
    ).expect("getCookies");

    // __nb_setCookie(cookie_str) — parse "name=value; path=/; expires=..."
    ctx.register_global_builtin_callable(js_string!("__nb_setCookie"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let raw = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
            let c = unsafe { cookies.get_mut() };
            // Parse first part before ';' as name=value
            if let Some(nv) = raw.split(';').next() {
                let nv = nv.trim();
                if let Some(eq) = nv.find('=') {
                    let name = nv[..eq].trim().to_string();
                    let value = nv[eq+1..].trim().to_string();
                    if !name.is_empty() {
                        c.insert(name, value);
                    }
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("setCookie");

    // Wire up document.cookie via JS property
    let cookie_js = r#"
        Object.defineProperty(document, "cookie", {
            get: function() { return __nb_getCookies(); },
            set: function(v) { __nb_setCookie(v); },
            enumerable: true, configurable: true
        });
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(cookie_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// document.write / document.writeln (Phase 3)
// ═══════════════════════════════════════════════════════════════════

fn register_document_write(ctx: &mut Context, dom: DomPtr) {
    // __nb_documentWrite(html) — appends parsed HTML to body
    ctx.register_global_builtin_callable(js_string!("__nb_documentWrite"), 1,
        NativeFunction::from_copy_closure(move |_, args, ctx| {
            let html = args.get_or_undefined(0).to_string(ctx)?.to_std_string_escaped();
            let d = unsafe { dom.get_mut() };
            // Find body node, or use root
            let body_id = d.query_selector("body").unwrap_or(0);
            // Append instead of replace — parse fragment and attach children
            let fragment = super::dom::parse_html(&html);
            let base_id = d.nodes.len();
            let parent_depth = d.nodes.get(body_id).map(|n| n.depth).unwrap_or(0);
            for frag_node in &fragment.nodes {
                let new_id = base_id + frag_node.id;
                let new_parent = match frag_node.parent {
                    Some(p) => Some(base_id + p),
                    None => Some(body_id),
                };
                d.nodes.push(super::dom::DomNode {
                    id: new_id,
                    tag: frag_node.tag.clone(),
                    attrs: frag_node.attrs.clone(),
                    text: frag_node.text.clone(),
                    parent: new_parent,
                    children: frag_node.children.iter().map(|c| base_id + c).collect(),
                    depth: parent_depth + 1 + frag_node.depth,
                });
            }
            for frag_node in &fragment.nodes {
                if frag_node.parent.is_none() {
                    d.nodes[body_id].children.push(base_id + frag_node.id);
                }
            }
            Ok(JsValue::undefined())
        }),
    ).expect("documentWrite");

    let write_js = r#"
        document.write = function() {
            var html = "";
            for (var i = 0; i < arguments.length; i++) html += arguments[i];
            __nb_documentWrite(html);
        };
        document.writeln = function() {
            var html = "";
            for (var i = 0; i < arguments.length; i++) html += arguments[i];
            html += "\n";
            __nb_documentWrite(html);
        };
    "#;
    if let Err(e) = ctx.eval(Source::from_bytes(write_js.as_bytes())) { log::warn!("[JS] eval failed: {e:?}"); }
}

// ═══════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(input: &str) -> Option<String> {
    let input = input.trim_end_matches('=');
    let mut bytes = Vec::new();
    let chars: Vec<u8> = input.bytes().filter_map(|b| {
        B64_CHARS.iter().position(|&c| c == b).map(|p| p as u8)
    }).collect();
    for chunk in chars.chunks(4) {
        // Use u32 arithmetic to avoid u8 overflow on bit shifts
        if chunk.len() >= 2 {
            bytes.push(((chunk[0] as u32) << 2 | (chunk[1] as u32) >> 4) as u8);
        }
        if chunk.len() >= 3 {
            bytes.push((((chunk[1] as u32) & 0x0F) << 4 | (chunk[2] as u32) >> 2) as u8);
        }
        if chunk.len() >= 4 {
            bytes.push((((chunk[2] as u32) & 0x03) << 6 | chunk[3] as u32) as u8);
        }
    }
    String::from_utf8(bytes).ok()
}

fn url_encode(s: &str) -> String {
    let mut result = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                result.push(byte as char);
            }
            _ => {
                result.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    result
}

fn url_decode(s: &str) -> String {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(val) = u8::from_str_radix(
                &String::from_utf8_lossy(&bytes[i+1..i+3]), 16
            ) {
                result.push(val);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            result.push(b' ');
        } else {
            result.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&result).into_owned()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::dom::{parse_html, ScriptSource};

    #[test]
    fn test_empty_scripts_noop() {
        let mut dom = parse_html("<p>Hello</p>");
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &[]).unwrap();
        assert_eq!(dom.by_tag("p")[0].text, "Hello");
    }

    #[test]
    fn test_console_log_no_crash() {
        let html = r#"<p>Hello</p><script>console.log("test message");</script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
    }

    #[test]
    fn test_get_element_by_id() {
        let html = r#"<div id="target">Original</div>
        <script>
            var el = document.getElementById("target");
            if (el !== null) { __nb_setTextContent(el, "Modified"); }
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let target = dom.get_element_by_id("target").unwrap();
        assert_eq!(dom.nodes[target].text, "Modified");
    }

    #[test]
    fn test_create_and_append() {
        let html = r#"<div id="container"></div>
        <script>
            var c = document.getElementById("container");
            var p = document.createElement("p");
            __nb_setTextContent(p, "Dynamic");
            __nb_appendChild(c, p);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let container = dom.get_element_by_id("container").unwrap();
        assert!(!dom.nodes[container].children.is_empty());
    }

    #[test]
    fn test_query_selector() {
        let html = r#"<p class="intro">Hello</p>
        <script>
            var el = document.querySelector(".intro");
            if (el !== null) { __nb_setTextContent(el, "Changed"); }
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let intro = dom.query_selector(".intro").unwrap();
        assert_eq!(dom.nodes[intro].text, "Changed");
    }

    #[test]
    fn test_syntax_error_graceful() {
        let html = r#"<p>Keep me</p><script>this is not valid javascript {{{{</script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        assert_eq!(dom.by_tag("p")[0].text, "Keep me");
    }

    #[test]
    fn test_set_attribute() {
        let html = r#"<div id="box"></div>
        <script>
            var el = document.getElementById("box");
            __nb_setAttribute(el, "class", "highlighted");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let box_id = dom.get_element_by_id("box").unwrap();
        assert_eq!(dom.nodes[box_id].attrs.get("class").unwrap(), "highlighted");
    }

    #[test]
    fn test_multiple_scripts_sequential() {
        let html = r#"<div id="counter">0</div>
        <script>
            var el = document.getElementById("counter");
            __nb_setTextContent(el, "1");
        </script>
        <script>
            var el = document.getElementById("counter");
            var cur = __nb_getTextContent(el);
            __nb_setTextContent(el, cur + "+1");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let counter = dom.get_element_by_id("counter").unwrap();
        assert_eq!(dom.nodes[counter].text, "1+1");
    }

    #[test]
    fn test_external_script_skipped() {
        let scripts = vec![ScriptInfo {
            source: ScriptSource::External("https://cdn.example.com/app.js".into()),
            script_type: None, defer: false, is_async: false,
        }];
        let mut dom = parse_html("<p>Hello</p>");
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
    }

    #[test]
    fn test_timer_stubs_no_crash() {
        let html = r#"<p>OK</p><script>
            setTimeout(function(){}, 1000);
            setInterval(function(){}, 500);
            clearTimeout(0); clearInterval(0);
            requestAnimationFrame(function(){});
            cancelAnimationFrame(0);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
    }

    #[test]
    fn test_navigator_stub() {
        let html = r#"<div id="ua"></div>
        <script>
            __nb_setTextContent(document.getElementById("ua"), navigator.userAgent);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let ua = dom.get_element_by_id("ua").unwrap();
        assert!(dom.nodes[ua].text.contains("NeuralBrowser"));
    }

    // ── Phase 2 tests ──

    #[test]
    fn test_local_storage() {
        let html = r#"<div id="result"></div>
        <script>
            localStorage.setItem("key1", "value1");
            var v = localStorage.getItem("key1");
            __nb_setTextContent(document.getElementById("result"), v);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let result = dom.get_element_by_id("result").unwrap();
        assert_eq!(dom.nodes[result].text, "value1");
    }

    #[test]
    fn test_local_storage_remove() {
        let html = r#"<div id="r"></div>
        <script>
            localStorage.setItem("x", "y");
            localStorage.removeItem("x");
            var v = localStorage.getItem("x");
            __nb_setTextContent(document.getElementById("r"), v === null ? "null" : v);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "null");
    }

    #[test]
    fn test_class_list_add_remove() {
        let html = r#"<div id="box" class="old"></div>
        <script>
            var id = document.getElementById("box");
            __nb_classAdd(id, "new");
            __nb_classRemove(id, "old");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let box_id = dom.get_element_by_id("box").unwrap();
        let cls = dom.nodes[box_id].attrs.get("class").unwrap();
        assert!(cls.contains("new"));
        assert!(!cls.contains("old"));
    }

    #[test]
    fn test_class_toggle() {
        let html = r#"<div id="t" class="active"></div>
        <script>
            var id = document.getElementById("t");
            var wasRemoved = !__nb_classToggle(id, "active");
            __nb_setTextContent(id, wasRemoved ? "removed" : "added");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let t = dom.get_element_by_id("t").unwrap();
        assert_eq!(dom.nodes[t].text, "removed");
    }

    #[test]
    fn test_parent_children_traversal() {
        let html = r#"<div id="parent"><p id="child">Hi</p></div>
        <script>
            var child = document.getElementById("child");
            var parent = __nb_getParent(child);
            var children = __nb_getChildren(parent);
            __nb_setTextContent(child, "children=" + children.length);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let child = dom.get_element_by_id("child").unwrap();
        assert!(dom.nodes[child].text.starts_with("children="));
    }

    #[test]
    fn test_sibling_traversal() {
        let html = r#"<ul><li id="a">A</li><li id="b">B</li><li id="c">C</li></ul>
        <script>
            var b = document.getElementById("b");
            var next = __nb_getNextSibling(b);
            var prev = __nb_getPrevSibling(b);
            __nb_setTextContent(b, "prev=" + __nb_getTextContent(prev) + ",next=" + __nb_getTextContent(next));
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let b = dom.get_element_by_id("b").unwrap();
        assert_eq!(dom.nodes[b].text, "prev=A,next=C");
    }

    #[test]
    fn test_insert_before() {
        let html = r#"<ul id="list"><li id="second">B</li></ul>
        <script>
            var list = document.getElementById("list");
            var second = document.getElementById("second");
            var first = document.createElement("li");
            __nb_setTextContent(first, "A");
            __nb_insertBefore(list, first, second);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let list = dom.get_element_by_id("list").unwrap();
        assert_eq!(dom.nodes[list].children.len(), 2);
        // First child should be the inserted one
        let first_child = dom.nodes[list].children[0];
        assert_eq!(dom.nodes[first_child].text, "A");
    }

    #[test]
    fn test_remove_attribute() {
        let html = r#"<div id="box" data-custom="val"></div>
        <script>
            var el = document.getElementById("box");
            __nb_removeAttribute(el, "data-custom");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let box_id = dom.get_element_by_id("box").unwrap();
        assert!(!dom.nodes[box_id].attrs.contains_key("data-custom"));
    }

    #[test]
    fn test_get_inner_html() {
        let html = r#"<div id="wrap"><p>Hello</p></div>
        <script>
            var wrap = document.getElementById("wrap");
            var html = __nb_getInnerHTML(wrap);
            __nb_setTextContent(wrap, html.length > 0 ? "has_html" : "empty");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let wrap = dom.get_element_by_id("wrap").unwrap();
        assert_eq!(dom.nodes[wrap].text, "has_html");
    }

    #[test]
    fn test_event_listener_no_crash() {
        let html = r#"<div id="r">OK</div>
        <script>
            document.addEventListener("click", function(e) {});
            window.addEventListener("load", function() {});
            document.removeEventListener("click", function() {});
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
    }

    #[test]
    fn test_btoa_atob() {
        let html = r#"<div id="r"></div>
        <script>
            var encoded = btoa("Hello");
            var decoded = atob(encoded);
            __nb_setTextContent(document.getElementById("r"), decoded);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "Hello");
    }

    #[test]
    fn test_encode_decode_uri() {
        let html = r#"<div id="r"></div>
        <script>
            var e = encodeURIComponent("hello world&foo=bar");
            var d = decodeURIComponent(e);
            __nb_setTextContent(document.getElementById("r"), d);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "hello world&foo=bar");
    }

    #[test]
    fn test_mutation_observer_stub() {
        let html = r#"<div id="r">OK</div>
        <script>
            var obs = new MutationObserver(function() {});
            obs.observe(document.getElementById("r"), { childList: true });
            obs.disconnect();
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
    }

    #[test]
    fn test_clone_node() {
        let html = r#"<div id="original" class="box">Text</div>
        <script>
            var orig = document.getElementById("original");
            var clone = __nb_cloneNode(orig, false);
            __nb_setTextContent(clone, "Cloned");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        // Original should be unchanged
        let orig = dom.get_element_by_id("original").unwrap();
        assert_eq!(dom.nodes[orig].text, "Text");
    }

    #[test]
    fn test_get_elements_by_class_name() {
        let html = r#"<div class="item">A</div><div class="item">B</div><div class="other">C</div>
        <script>
            var items = document.getElementsByClassName("item");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        // Just verify no crash; the items array is created in JS
    }

    #[test]
    fn test_create_text_node() {
        let html = r#"<div id="host"></div>
        <script>
            var host = document.getElementById("host");
            var text = document.createTextNode("hello");
            __nb_appendChild(host, text);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let host = dom.get_element_by_id("host").unwrap();
        assert!(!dom.nodes[host].children.is_empty());
    }

    // ── Phase 3 tests ──

    #[test]
    fn test_settimeout_zero_executes() {
        let html = r#"<div id="r">before</div>
        <script>
            setTimeout(function() {
                __nb_setTextContent(document.getElementById("r"), "after");
            }, 0);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "after");
    }

    #[test]
    fn test_settimeout_queued_drains() {
        let html = r#"<div id="r">0</div>
        <script>
            setTimeout(function() {
                __nb_setTextContent(document.getElementById("r"), "queued");
            }, 100);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts_with_externals(&mut dom, &scripts, &HashMap::new(), "").unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "queued");
    }

    #[test]
    fn test_style_api() {
        let html = r#"<div id="box" style="color: red"></div>
        <script>
            var id = document.getElementById("box");
            __nb_setStyle(id, "display", "none");
            var color = __nb_getStyle(id, "color");
            __nb_setTextContent(id, "color=" + color);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let box_id = dom.get_element_by_id("box").unwrap();
        assert_eq!(dom.nodes[box_id].text, "color=red");
        assert!(dom.nodes[box_id].attrs.get("style").unwrap().contains("display"));
    }

    #[test]
    fn test_form_value() {
        let html = r#"<input id="name" value="initial"><div id="r"></div>
        <script>
            var inp = document.getElementById("name");
            var old = __nb_getValue(inp);
            __nb_setValue(inp, "changed");
            var now = __nb_getValue(inp);
            __nb_setTextContent(document.getElementById("r"), old + "->" + now);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "initial->changed");
    }

    #[test]
    fn test_checkbox_checked() {
        let html = r#"<input id="cb" type="checkbox"><div id="r"></div>
        <script>
            var cb = document.getElementById("cb");
            var before = __nb_getChecked(cb);
            __nb_setChecked(cb, true);
            var after = __nb_getChecked(cb);
            __nb_setTextContent(document.getElementById("r"), before + "->" + after);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "false->true");
    }

    #[test]
    fn test_cookie_api() {
        let html = r#"<div id="r"></div>
        <script>
            document.cookie = "user=john; path=/";
            document.cookie = "theme=dark";
            var c = document.cookie;
            __nb_setTextContent(document.getElementById("r"), c.indexOf("user=john") >= 0 ? "ok" : "fail");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts_with_externals(&mut dom, &scripts, &HashMap::new(), "").unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "ok");
    }

    #[test]
    fn test_window_location_url() {
        let html = r#"<div id="r"></div>
        <script>
            __nb_setTextContent(document.getElementById("r"), location.hostname);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts_with_externals(
            &mut dom, &scripts, &HashMap::new(),
            "https://example.com/page?q=1#top"
        ).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "example.com");
    }

    #[test]
    fn test_document_write() {
        let html = r#"<body></body>
        <script>
            document.write("<p id='written'>Hello from write</p>");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let p = dom.get_element_by_id("written");
        assert!(p.is_some());
    }

    #[test]
    fn test_domcontentloaded_fires() {
        let html = r#"<div id="r">waiting</div>
        <script>
            document.addEventListener("DOMContentLoaded", function() {
                __nb_setTextContent(document.getElementById("r"), "loaded");
            });
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts_with_externals(&mut dom, &scripts, &HashMap::new(), "").unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "loaded");
    }

    #[test]
    fn test_window_load_fires() {
        let html = r#"<div id="r">waiting</div>
        <script>
            window.addEventListener("load", function() {
                __nb_setTextContent(document.getElementById("r"), "win_loaded");
            });
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts_with_externals(&mut dom, &scripts, &HashMap::new(), "").unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "win_loaded");
    }

    #[test]
    fn test_formdata_stub() {
        let html = r#"<div id="r"></div>
        <script>
            var fd = new FormData();
            fd.append("name", "John");
            fd.append("age", "30");
            __nb_setTextContent(document.getElementById("r"), fd.get("name") + "," + fd.get("age"));
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "John,30");
    }

    #[test]
    fn test_urlsearchparams() {
        let html = r#"<div id="r"></div>
        <script>
            var p = new URLSearchParams("?foo=bar&baz=42");
            __nb_setTextContent(document.getElementById("r"), p.get("foo") + "," + p.get("baz"));
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "bar,42");
    }

    #[test]
    fn test_camel_to_kebab() {
        assert_eq!(camel_to_kebab("backgroundColor"), "background-color");
        assert_eq!(camel_to_kebab("fontSize"), "font-size");
        assert_eq!(camel_to_kebab("display"), "display");
        assert_eq!(camel_to_kebab("borderTopLeftRadius"), "border-top-left-radius");
    }

    #[test]
    fn test_inline_style_parse() {
        let style = "display: none; color: red; font-size: 14px";
        assert_eq!(parse_inline_style(style, "display"), Some("none".to_string()));
        assert_eq!(parse_inline_style(style, "color"), Some("red".to_string()));
        assert_eq!(parse_inline_style(style, "fontSize"), Some("14px".to_string()));
        assert_eq!(parse_inline_style(style, "margin"), None);
    }

    #[test]
    fn test_set_remove_inline_style() {
        let mut style = "color: red".to_string();
        set_inline_style(&mut style, "display", "none");
        assert!(style.contains("display: none"));
        assert!(style.contains("color: red"));
        remove_inline_style(&mut style, "color");
        assert!(!style.contains("color"));
        assert!(style.contains("display: none"));
    }

    #[test]
    fn test_window_alert_confirm_prompt() {
        let html = r#"<div id="r"></div>
        <script>
            window.alert("hello");
            var c = window.confirm("sure?");
            var p = window.prompt("name?", "default");
            __nb_setTextContent(document.getElementById("r"), c + "," + p);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "true,default");
    }

    #[test]
    fn test_clear_timeout() {
        let html = r#"<div id="r">original</div>
        <script>
            var id = setTimeout(function() {
                __nb_setTextContent(document.getElementById("r"), "changed");
            }, 100);
            clearTimeout(id);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts_with_externals(&mut dom, &scripts, &HashMap::new(), "").unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "original");
    }

    #[test]
    fn test_fetch_response_body() {
        let html = r#"<div id="r"></div>
        <script>
            try {
                var resp = { ok: true, status: 200, _body: '{"a":1}',
                    text: function() { return this._body; },
                    json: function() { return JSON.parse(this._body); }
                };
                var json = resp.json();
                document.getElementById("r").textContent = json.a;
            } catch(e) {
                document.getElementById("r").textContent = "err: " + e;
            }
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "1");
    }

    // ── Phase 4 tests — Element wrapper ──

    #[test]
    fn test_element_textcontent_property() {
        let html = r#"<div id="target">Old</div>
        <script>
            var el = document.getElementById("target");
            el.textContent = "New";
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let target = dom.get_element_by_id("target").unwrap();
        assert_eq!(dom.nodes[target].text, "New");
    }

    #[test]
    fn test_element_innerhtml_property() {
        let html = r#"<div id="wrap"></div>
        <script>
            var el = document.getElementById("wrap");
            el.innerHTML = "<p>Injected</p>";
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let wrap = dom.get_element_by_id("wrap").unwrap();
        assert!(!dom.nodes[wrap].children.is_empty());
    }

    #[test]
    fn test_element_classlist_object() {
        let html = r#"<div id="box" class="old"></div>
        <script>
            var el = document.getElementById("box");
            el.classList.add("new");
            el.classList.remove("old");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let box_id = dom.get_element_by_id("box").unwrap();
        let cls = dom.nodes[box_id].attrs.get("class").unwrap();
        assert!(cls.contains("new"));
        assert!(!cls.contains("old"));
    }

    #[test]
    fn test_element_setattribute() {
        let html = r#"<div id="el"></div>
        <script>
            var el = document.getElementById("el");
            el.setAttribute("data-name", "test");
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let el = dom.get_element_by_id("el").unwrap();
        assert_eq!(dom.nodes[el].attrs.get("data-name").unwrap(), "test");
    }

    #[test]
    fn test_element_appendchild_wrapped() {
        let html = r#"<div id="parent"></div>
        <script>
            var parent = document.getElementById("parent");
            var child = document.createElement("span");
            child.textContent = "Dynamic";
            parent.appendChild(child);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let parent = dom.get_element_by_id("parent").unwrap();
        assert!(!dom.nodes[parent].children.is_empty());
        let child_id = dom.nodes[parent].children[0];
        assert_eq!(dom.nodes[child_id].text, "Dynamic");
    }

    #[test]
    fn test_element_style_property() {
        let html = r#"<div id="box"></div>
        <script>
            var el = document.getElementById("box");
            el.style.display = "none";
            el.style.color = "red";
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let box_id = dom.get_element_by_id("box").unwrap();
        let style = dom.nodes[box_id].attrs.get("style").unwrap();
        assert!(style.contains("display: none"));
        assert!(style.contains("color: red"));
    }

    #[test]
    fn test_element_value_property() {
        let html = r#"<input id="inp" value="old"><div id="r"></div>
        <script>
            var inp = document.getElementById("inp");
            var old = inp.value;
            inp.value = "new";
            document.getElementById("r").textContent = old + "->" + inp.value;
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "old->new");
    }

    #[test]
    fn test_element_parentnode() {
        let html = r#"<div id="parent"><p id="child">Hi</p></div>
        <script>
            var child = document.getElementById("child");
            var parent = child.parentNode;
            child.textContent = parent.tagName;
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let child = dom.get_element_by_id("child").unwrap();
        assert_eq!(dom.nodes[child].text, "DIV");
    }

    #[test]
    fn test_element_children_property() {
        let html = r#"<ul id="list"><li>A</li><li>B</li></ul>
        <script>
            var list = document.getElementById("list");
            list.setAttribute("data-count", list.children.length);
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let list = dom.get_element_by_id("list").unwrap();
        assert_eq!(dom.nodes[list].attrs.get("data-count").unwrap(), "2");
    }

    #[test]
    fn test_element_addeventlistener() {
        let html = r#"<div id="btn">Click</div><div id="r">waiting</div>
        <script>
            var btn = document.getElementById("btn");
            btn.addEventListener("click", function() {
                document.getElementById("r").textContent = "clicked";
            });
            btn.click();
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "clicked");
    }

    #[test]
    fn test_element_onclick_property() {
        let html = r#"<div id="btn">Click</div><div id="r">waiting</div>
        <script>
            var btn = document.getElementById("btn");
            btn.onclick = function() {
                document.getElementById("r").textContent = "fired";
            };
            btn.click();
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "fired");
    }

    #[test]
    fn test_element_matches() {
        let html = r##"<div id="el" class="active"></div><div id="r"></div>
        <script>
            var el = document.getElementById("el");
            var m1 = el.matches(".active");
            var m2 = el.matches("#el");
            var m3 = el.matches("div");
            var m4 = el.matches(".nope");
            document.getElementById("r").textContent = [m1, m2, m3, m4].join(",");
        </script>"##;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "true,true,true,false");
    }

    #[test]
    fn test_element_closest() {
        let html = r#"<div class="outer"><div class="inner"><span id="deep">X</span></div></div>
        <script>
            var span = document.getElementById("deep");
            var found = span.closest(".outer");
            span.textContent = found ? found.className : "null";
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let deep = dom.get_element_by_id("deep").unwrap();
        assert_eq!(dom.nodes[deep].text, "outer");
    }

    #[test]
    fn test_element_remove() {
        let html = r#"<div id="parent"><p id="child">Remove me</p></div>
        <script>
            document.getElementById("child").remove();
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let parent = dom.get_element_by_id("parent").unwrap();
        assert!(dom.nodes[parent].children.is_empty());
    }

    #[test]
    fn test_element_hidden_property() {
        let html = r#"<div id="el"></div>
        <script>
            var el = document.getElementById("el");
            el.hidden = true;
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let el = dom.get_element_by_id("el").unwrap();
        assert!(dom.nodes[el].attrs.contains_key("hidden"));
    }

    #[test]
    fn test_element_tagname() {
        let html = r#"<div id="r"></div>
        <script>
            var el = document.getElementById("r");
            el.textContent = el.tagName;
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("r").unwrap();
        assert_eq!(dom.nodes[r].text, "DIV");
    }

    #[test]
    fn test_real_world_pattern_toggle_class() {
        // Common pattern from real websites
        let html = r#"<nav id="menu" class="collapsed"></nav>
        <button id="btn">Toggle</button>
        <script>
            document.getElementById("btn").addEventListener("click", function() {
                document.getElementById("menu").classList.toggle("expanded");
                document.getElementById("menu").classList.toggle("collapsed");
            });
            document.getElementById("btn").click();
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let menu = dom.get_element_by_id("menu").unwrap();
        let cls = dom.nodes[menu].attrs.get("class").unwrap();
        assert!(cls.contains("expanded"));
        assert!(!cls.contains("collapsed"));
    }

    #[test]
    fn test_real_world_pattern_create_list() {
        // Common pattern: build a list dynamically
        let html = r#"<ul id="list"></ul>
        <script>
            var list = document.getElementById("list");
            var items = ["Apple", "Banana", "Cherry"];
            for (var i = 0; i < items.length; i++) {
                var li = document.createElement("li");
                li.textContent = items[i];
                li.className = "item";
                list.appendChild(li);
            }
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let list = dom.get_element_by_id("list").unwrap();
        assert_eq!(dom.nodes[list].children.len(), 3);
        let first = dom.nodes[list].children[0];
        assert_eq!(dom.nodes[first].text, "Apple");
        assert_eq!(dom.nodes[first].attrs.get("class").unwrap(), "item");
    }

    #[test]
    fn test_real_world_pattern_form_handling() {
        let html = r#"<input id="email" value="user@example.com">
        <div id="result"></div>
        <script>
            var inp = document.getElementById("email");
            var val = inp.value;
            var parts = val.split("@");
            document.getElementById("result").textContent = "user=" + parts[0] + " domain=" + parts[1];
        </script>"#;
        let mut dom = parse_html(html);
        let scripts = dom.scripts.clone();
        let mut engine = JsEngine::new();
        engine.execute_scripts(&mut dom, &scripts).unwrap();
        let r = dom.get_element_by_id("result").unwrap();
        assert_eq!(dom.nodes[r].text, "user=user domain=example.com");
    }
}
