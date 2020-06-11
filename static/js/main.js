function validate_filetype() {
    var fileName = document.getElementById('fileupload').value
    var allowed_extensions = new Array("jpg", "png", "bmp", "jpeg");
    var file_extension = fileName.split('.').pop().toLowerCase();

    for (var i = 0; i < allowed_extensions.length; i++) {
        if (allowed_extensions[i] == file_extension) {
            return true;
        }
    }
    alert('invalid file type')
    return false;
}