use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')";

    let Ok(output) = Command::new(&python).arg("-c").arg(script).output() else {
        return;
    };
    if !output.status.success() {
        return;
    }

    let Ok(libdir) = String::from_utf8(output.stdout) else {
        return;
    };
    let libdir = libdir.trim();
    if libdir.is_empty() {
        return;
    }

    println!("cargo:rustc-link-search=native={libdir}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
}
